#%%
# 1. 라이브러리 임포트 및 API 키 설정
###############################################################################################################################################################################

import os
import google.generativeai as genai
import json
import argparse
import time
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions

# Import prompts from external file
import prompts

# Configuration
MODEL_NAME = "models/gemini-2.5-pro"
BUFFER_SIZE = 10             # 앞뒤 문맥 포함 개수
WAIT_SECONDS = 60            # API 호출 간 대기 시간 (초)

# ==============================================================================
# 1. API 호출 헬퍼 함수 (Retry 적용)
# ==============================================================================
@retry(
    retry=retry_if_exception_type((
        exceptions.ResourceExhausted, 
        exceptions.ServiceUnavailable, 
        exceptions.GoogleAPICallError,
        exceptions.InternalServerError
    )),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_content_with_retry(model, prompt):
    """
    Gemini API 호출을 수행하며, 실패 시 지수 백오프로 재시도합니다.
    """
    return model.generate_content(prompt)

# ==============================================================================
# 2. ID 기반 텍스트 추출 함수 (Buffer 적용)
# ==============================================================================
def get_transcript_segment(all_utterances, start_id, end_id, buffer=5):
    """
    전체 발화 목록에서 특정 ID 구간의 텍스트만 추출합니다.
    앞뒤로 buffer만큼의 발화를 더 포함하여 문맥을 확보합니다.
    """
    start_idx = -1
    end_idx = -1
    
    # 인덱스 찾기 (문자열/정수 호환 위해 str 변환 비교)
    for i, u in enumerate(all_utterances):
        if str(u.get('id')) == str(start_id):
            start_idx = i
        if str(u.get('id')) == str(end_id):
            end_idx = i
            if start_idx != -1: break 
            
    if start_idx == -1 or end_idx == -1:
        return "" # ID를 못 찾은 경우 빈 문자열 반환

    # 버퍼 적용 (리스트 범위 보호)
    real_start = max(0, start_idx - buffer)
    real_end = min(len(all_utterances), end_idx + 1 + buffer)
    
    segment_lines = []
    for i in range(real_start, real_end):
        u = all_utterances[i]
        u_id = u.get('id')
        name = u.get('name', 'Unknown')
        content = u.get('content', '')
        segment_lines.append(f"[ID: {u_id}] {name}: {content}")
        
    return "\n".join(segment_lines)


# ==============================================================================
# 3. 구조 분석 함수 (Step 2 & 3)
# ==============================================================================
def analyze_structure(file_id):
    print(f"\n{'='*80}")
    print(f"🏗️ [Step 1] 구조 분석 시작: {file_id}")
    print(f"{'='*80}\n")
    
    json_file_path = f"{file_id}_cleansed.json"
    
    try:
        # 1. JSON 파일 직접 읽기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            meeting_log_data = json.load(f)
        
        # --- 프롬프트에 포함할 내용 가공 ---
        metadata_str = json.dumps(meeting_log_data.get('metadata', {}), ensure_ascii=False, indent=2)
        speakers_str = json.dumps(meeting_log_data.get('speakers', []), ensure_ascii=False, indent=2)
        
        speaker_map = {speaker['id']: speaker.get('name', f"P{i:02d}") for i, speaker in enumerate(meeting_log_data.get('speakers', []))}
        conversation_text_lines = []
        
        for utterance in meeting_log_data.get('utterances', []):
            u_id = utterance.get('id')
            speaker_id = utterance.get('name') 
            message = utterance.get('content')
            
            if speaker_id and message and u_id:
                speaker_label = speaker_map.get(speaker_id, speaker_id)
                conversation_text_lines.append(f"[ID: {u_id}] {speaker_label}: {message}")
                
        conversation_text = "\n".join(conversation_text_lines)
        
        prompt_input_text = f"""# Metadata
{metadata_str}

# Speakers
{speakers_str}

# Conversation
{conversation_text}
"""
        print(f"'{json_file_path}' 파일 내용 로드 및 프롬프트용 데이터 가공 완료.")

        # --- 구조 및 구간 추출용 프롬프트 ---
        # prompts.py에서 템플릿 가져오기
        prompt_text_template = prompts.STRUCTURE_PROMPT.format(input_data=prompt_input_text)

        # 모델 초기화
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini 모델 초기화 성공. (모델: {MODEL_NAME})")
        
        # 프롬프트 전달 (Retry 적용)
        print("---전송될 프롬프트---")
        response = generate_content_with_retry(model, prompt_text_template)
        
        print("\n--- Gemini API 응답 ---")
        
        # JSON 파싱
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(json_string)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        
        # Skeleton 저장
        if 'skeleton' not in meeting_log_data:
            meeting_log_data['skeleton'] = {}
            
        meeting_log_data['skeleton']['main_topic'] = parsed_json.get('main_topic', '')
        meeting_log_data['skeleton']['domain'] = parsed_json.get('domain', '')
        meeting_log_data['skeleton']['topics'] = parsed_json.get('topics', [])
        
        output_json_file_path = f"{file_id}_step1.json" 
        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(meeting_log_data, f, ensure_ascii=False, indent=4)
            
        print(f"  파일 저장 완료: '{output_json_file_path}'")
        return True

    except FileNotFoundError:
        print(f"오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"구조 분석 중 오류 발생: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# 4. 상세 분석 및 통합 함수 (Step 4 & 5)
# ==============================================================================
def analyze_details_and_consolidate(file_id):
    print(f"\n{'='*80}")
    print(f"🔍 [Step 2] 상세 분석 및 통합 시작: {file_id}")
    print(f"{'='*80}\n")
    
    step1_file_path = f"{file_id}_step1.json"
    
    try:
        with open(step1_file_path, 'r', encoding='utf-8') as f:
            meeting_log_data = json.load(f)
            
        topics_list = meeting_log_data.get('skeleton', {}).get('topics', [])
        all_utterances = meeting_log_data.get('utterances', [])
        final_topics = []
        
        if not topics_list:
            print("⚠️ 처리할 토픽이 없습니다. Step 1 결과를 확인하세요.")
            return False
            
        total_topics = len(topics_list)
        print(f"✅ 총 {total_topics}개의 토픽을 분석합니다. (Buffer: ±{BUFFER_SIZE}, 대기시간: {WAIT_SECONDS}초)\n")
        
        model = genai.GenerativeModel(MODEL_NAME)
        
        # --- 상세 분석 Loop ---
        for index, topic_item in enumerate(topics_list):
            topic_item['sub_topic_id'] = str(index + 1)
            sub_topic = topic_item.get('sub_topic', '제목 없음')
            topic_type = topic_item.get('type', 'unknown')
            start_id = topic_item.get('start_id')
            end_id = topic_item.get('end_id')
            
            print(f"🔄 [Topic {index+1}/{total_topics}] 처리 중...")
            print(f"   - 주제: {sub_topic}")
            print(f"   - 유형: {topic_type}")
            print(f"   - 구간: ID {start_id} ~ {end_id}")
            
            segment_text = get_transcript_segment(all_utterances, start_id, end_id, buffer=BUFFER_SIZE)
            
            if not segment_text:
                print("   -> 경고: 텍스트 추출 실패 (ID 확인 필요). Skip.")
                topic_item['error'] = "Text extraction failed"
                final_topics.append(topic_item)
                continue

            # prompts.py에서 템플릿 가져오기
            type_instruction = prompts.TYPE_PROMPTS.get(topic_type, prompts.DEFAULT_PROMPT)
            
            step2_prompt = f"""
# 페르소나
당신은 회의록의 특정 세그먼트를 정밀 분석하는 전문가입니다.

# 작업 개요
* **분석 대상 주제**: '{sub_topic}'
* **핵심 논의 구간**: ID {start_id}번 ~ {end_id}번 발화
* **참고 문맥(Buffer)**: 핵심 구간의 앞뒤로 각각 {BUFFER_SIZE}개의 발화가 문맥 파악을 위해 추가되었습니다.

# 작업 지시
제공된 [대화 내용]을 읽고, **핵심 논의 구간**을 중심으로 다음 내용을 추출하세요.

1. **details (상세 내용)**: 아래 [작성 지침]에 정의된 구조대로 작성하세요.
2. **segment_decisions (결정 사항)**: 이 구간에서 확정된 합의나 결정 사항이 있다면 명확한 문장으로 추출하세요. (없으면 빈 리스트)
3. **segment_action_items (실행 항목)**: 구체적인 할 일(Task), 담당자(Assignee), 기한(Due Date)을 추출하세요. (없으면 빈 리스트)

# 작성 지침 (JSON Schema & Guide)
{type_instruction}

# 필수 출력 형식 (JSON Only)
반드시 아래 포맷으로 응답하세요. 마크다운(```json)이나 추가 설명은 제외하세요.
{{
  "short_summary": "이 주제에 대한 1~2문장 요약",
  "details": {{ ...위 작성 지침의 구조... }},
  "segment_decisions": [
    "결정된 사항 1",
    "결정된 사항 2"
  ],
  "segment_action_items": [
    {{
      "task": "구체적인 작업 내용",
      "assignee": "담당자 (또는 '미정')",
      "due_date": "마감기한 (또는 '미정')"
    }}
  ]
}}

# 대화 내용
{segment_text}
"""
            try:
                print("   -> API 호출 중...")
                # Retry 적용된 함수 호출
                response = generate_content_with_retry(model, step2_prompt)
                
                json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
                parsed_response = json.loads(json_string)
                
                topic_item.update(parsed_response)
                print("   -> 분석 및 병합 완료")
            except Exception as e:
                print(f"   -> API 호출/파싱 오류: {e}")
                topic_item['error'] = str(e)
            
            final_topics.append(topic_item)

            if index < total_topics - 1:
                print(f"   -> API 제한 준수를 위해 {WAIT_SECONDS}초 대기합니다...")
                time.sleep(WAIT_SECONDS)
        
        # Step 2 결과 저장
        meeting_log_data['skeleton']['topics'] = final_topics
        output_file_path = f"{file_id}_step2.json"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(meeting_log_data, f, ensure_ascii=False, indent=2)
        print(f"\n Step 2 완료! 결과가 저장되었습니다: {output_file_path}")
        
        # --- 최종 통합 (Consolidation) ---
        print(f"✅ 분석된 토픽 {len(final_topics)}개를 바탕으로 최종 요약을 시작합니다.")
        
        topics_json_str = json.dumps(final_topics, ensure_ascii=False, indent=2)
        # prompts.py에서 템플릿 가져오기
        final_prompt = prompts.CONSOLIDATION_PROMPT.format(topics_json=topics_json_str)
        
        print("🚀 Gemini API 호출 중... (Final Consolidation)")
        # Retry 적용된 함수 호출
        response = generate_content_with_retry(model, final_prompt)
        
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_result = json.loads(json_string)
        
        ordered_summary = {}
        ordered_summary['main_topic'] = meeting_log_data['skeleton'].get('main_topic', '')
        ordered_summary['domain'] = meeting_log_data['skeleton'].get('domain', '')
        ordered_summary['summary'] = parsed_result.get('summary', '')
        ordered_summary['decisions'] = parsed_result.get('decisions', [])
        ordered_summary['action_items'] = parsed_result.get('action_items', [])
        ordered_summary['topics'] = final_topics
        
        final_output_data = {
            "metadata": meeting_log_data.get('metadata', {}),
            "final_summary": ordered_summary,
            "speakers": meeting_log_data.get('speakers', []),
            "utterances": meeting_log_data.get('utterances', [])
        }
        
        final_output_path = f"{file_id}_final.json"
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n🎉 [최종 완료] 회의록 생성이 끝났습니다!")
        print(f"💾 파일 저장 경로: {final_output_path}")
        return True

    except FileNotFoundError:
        print(f"오류: '{step1_file_path}' 파일을 찾을 수 없습니다. Step 1을 먼저 실행하세요.")
        return False
    except Exception as e:
        print(f"상세 분석 중 오류 발생: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # API 키 설정
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        print("Gemini API 키 설정 완료.")
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        exit(1)

    # Argument Parsing
    parser = argparse.ArgumentParser(description="Gemini API Test Script")
    parser.add_argument("--file_ids", nargs='+', required=True, help="Target File IDs (space separated)")
    parser.add_argument("--mode", choices=['all', 'structure', 'details'], default='all', help="Execution mode: 'all' (default), 'structure' (Step 1 only), 'details' (Step 2 only)")
    args, _ = parser.parse_known_args()
    
    target_file_ids = args.file_ids
    mode = args.mode
    
    print(f"총 {len(target_file_ids)}개의 파일을 처리합니다: {target_file_ids}")
    print(f"실행 모드: {mode}")
    
    for file_id in target_file_ids:
        # Mode: structure (Step 1 only)
        if mode == 'structure':
            analyze_structure(file_id)
            
        # Mode: details (Step 2 only - requires previous step)
        elif mode == 'details':
            analyze_details_and_consolidate(file_id)
            
        # Mode: all (Step 1 -> Step 2)
        else:
            success_step1 = analyze_structure(file_id)
            if success_step1:
                analyze_details_and_consolidate(file_id)
            else:
                print(f"⛔ {file_id}: 구조 분석 실패로 인해 상세 분석을 건너뜁니다.")
