#%%
# 1. 라이브러리 임포트 및 API 키 설정
###############################################################################################################################################################################

import os
import google.generativeai as genai
import json
import argparse
import time
import traceback
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions
from dotenv import load_dotenv
import requests
import sys

# Import prompts from external file
import prompts

load_dotenv()

# Configuration
MODEL_NAME = "models/gemini-2.5-pro"
BUFFER_SIZE = 10             # 앞뒤 문맥 포함 개수
WAIT_SECONDS = 31            # API 호출 간 대기 시간 (초)
BUCKET_NAME = "hedj-s3-1"    # S3 버킷 이름

# S3 Client 초기화
s3_client = boto3.client('s3')


def update_session_status(room_name, status):
    """
    방 이름(room_name)으로 세션 상태를 업데이트합니다.
    status: "BEFORE_START", "IN_PROGRESS", "COMPLETED" 중 하나
    """
    # URL에서 ID가 빠지고 /status로 변경됨
    url = "http://localhost:8080/api/sessions/status"

    headers = {
        "Content-Type": "application/json"
    }

    # Body에 roomName 포함
    data = {
        "roomName": room_name,
        "status": status
    }

    try:
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            print(f"성공: 방 '{room_name}'의 상태가 {status}로 변경되었습니다.")
        else:
            print(f"실패: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"에러 발생: {e}")

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
        name = u.get('USER_ID', 'Unknown')
        content = u.get('content', '')
        segment_lines.append(f"[ID: {u_id}] {name}: {content}")
        
    return "\n".join(segment_lines)


# ==============================================================================
# 3. 구조 분석 함수 (Step 1)
# ==============================================================================
def analyze_structure(file_id):
    print(f"\n{'='*80}")
    print(f"🏗️ [Step 1] 구조 분석 시작: {file_id}")
    print(f"{'='*80}\n")
    
    # S3 경로 설정
    input_s3_key = f"meeting_logs/{file_id}.json"
    
    try:
        # 1. S3에서 JSON 파일 읽기
        print(f"S3에서 파일 읽는 중: s3://{BUCKET_NAME}/{input_s3_key}")
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=input_s3_key)
        meeting_log_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # --- 프롬프트에 포함할 내용 가공 ---
        metadata_str = json.dumps(meeting_log_data.get('metadata', {}), ensure_ascii=False, indent=2)
        participants_str = json.dumps(meeting_log_data.get('participants', []), ensure_ascii=False, indent=2)
        
        speaker_map = {p['USER_ID']: p.get('name', f"P{i:02d}") for i, p in enumerate(meeting_log_data.get('participants', []))}
        conversation_text_lines = []
        
        for utterance in meeting_log_data.get('utterances', []):
            u_id = utterance.get('id')
            speaker_id = utterance.get('USER_ID') 
            message = utterance.get('content')
            
            if speaker_id and message and u_id:
                speaker_label = speaker_map.get(speaker_id, speaker_id)
                conversation_text_lines.append(f"[ID: {u_id}] {speaker_label}: {message}")
                
        conversation_text = "\n".join(conversation_text_lines)
        
        prompt_input_text = f"""# Metadata
{metadata_str}

# Participants
{participants_str}

# Conversation
{conversation_text}
"""
        print(f"S3 파일 내용 로드 및 프롬프트용 데이터 가공 완료.")

        # --- 구조 및 구간 추출용 프롬프트 ---
        # prompts.py에서 템플릿 가져오기
        prompt_text_template = prompts.STRUCTURE_PROMPT.format(input_data=prompt_input_text)

        # 모델 초기화
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini 모델 초기화 성공. (모델: {MODEL_NAME})")
        
        # 프롬프트 전달 (Retry 적용)
        print("---Gemini API 호출 중---")
        response = generate_content_with_retry(model, prompt_text_template)
        
        print("\n--- Gemini API 응답 ---")
        
        # JSON 파싱
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(json_string)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        
        # Skeleton 저장 (메모리)
        if 'skeleton' not in meeting_log_data:
            meeting_log_data['skeleton'] = {}
            
        meeting_log_data['skeleton']['main_topic'] = parsed_json.get('main_topic', '')
        meeting_log_data['skeleton']['domain'] = parsed_json.get('domain', '')
        meeting_log_data['skeleton']['topics'] = parsed_json.get('topics', [])
        
        print(f"  Step 1 완료 (메모리에 저장)")
        return meeting_log_data  # 데이터 반환

    except s3_client.exceptions.NoSuchKey:
        print(f"오류: S3에서 '{input_s3_key}' 파일을 찾을 수 없습니다.")
        return None
    except ClientError as e:
        print(f"S3 오류 발생: {e}")
        return None
    except Exception as e:
        print(f"구조 분석 중 오류 발생: {e}")
        traceback.print_exc()
        return None


# ==============================================================================
# 4. 상세 분석 및 통합 함수 (Step 2)
# ==============================================================================
def analyze_details_and_consolidate(file_id, meeting_log_data):
    print(f"\n{'='*80}")
    print(f"🔍 [Step 2] 상세 분석 및 통합 시작: {file_id}")
    print(f"{'='*80}\n")
    
    # S3 경로 설정 (최종 결과만)
    final_s3_key = f"Summarize/{file_id}_final.json"
    
    try:
        topics_list = meeting_log_data.get('skeleton', {}).get('topics', [])
        all_utterances = meeting_log_data.get('utterances', [])
        participants = meeting_log_data.get('participants', [])
        final_topics = []
        
        # Participants 정보를 JSON 문자열로 변환
        participants_info = json.dumps(participants, ensure_ascii=False, indent=2)
        
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

# 참가자 정보
다음은 회의 참가자 목록입니다. action_items의 assignee를 지정할 때 **반드시** 이 정보를 참고하세요.
{participants_info}

# 작업 지시
제공된 [대화 내용]을 읽고, **핵심 논의 구간**을 중심으로 다음 내용을 추출하세요.

1. **details (상세 내용)**: 아래 [작성 지침]에 정의된 구조대로 작성하세요.
2. **segment_decisions (결정 사항)**: 이 구간에서 확정된 합의나 결정 사항이 있다면 명확한 문장으로 추출하세요. (없으면 빈 리스트)
3. **segment_action_items (실행 항목)**: 구체적인 할 일(Task), 담당자(Assignee), 기한(Due Date)을 추출하세요.
   - **중요:** assignee는 반드시 위 [참가자 정보]의 'name' 필드 값을 사용하세요. USER_ID를 사용하지 마세요.
   - 대화에서 "제가 할게요" 같은 표현이 나오면, 해당 발화자의 USER_ID를 확인하고 [참가자 정보]에서 매칭되는 name을 찾아 사용하세요.
   - 담당자가 불명확하거나 [참가자 정보]에서 찾을 수 없으면 '미정'으로 표기하세요.
   - (없으면 빈 리스트)

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
      "assignee": "담당자 이름 (또는 '미정')",
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
        
        # --- 최종 통합 (Consolidation) ---
        print(f"\n✅ 분석된 토픽 {len(final_topics)}개를 바탕으로 최종 요약을 시작합니다.")
        
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
            "participants": meeting_log_data.get('participants', []),
            "utterances": meeting_log_data.get('utterances', [])
        }
        
        # 최종 결과 저장 (S3)
        print(f"\nS3에 최종 파일 저장 중: s3://{BUCKET_NAME}/{final_s3_key}")
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=final_s3_key,
            Body=json.dumps(final_output_data, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
            
        print(f"🎉 [최종 완료] 회의록 생성이 끝났습니다!")
        print(f"💾 파일 저장 경로: s3://{BUCKET_NAME}/{final_s3_key}")
        sys.exit(0)
        #return True

    except Exception as e:
        print(f"상세 분석 중 오류 발생: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # API 키 설정
    try:
        api_key = "-"
        if api_key is None:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        print("Gemini API 키 설정 완료.")
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        exit(1)

    # Argument Parsing
    parser = argparse.ArgumentParser(description="S3 기반 Gemini API 회의록 요약 스크립트")
    parser.add_argument("--file_ids", nargs='+', required=True, help="Target File IDs (space separated, e.g., 'room001_20231121_143000')")
    args, _ = parser.parse_known_args()
    
    target_file_ids = args.file_ids
    
    print(f"총 {len(target_file_ids)}개의 파일을 처리합니다: {target_file_ids}")
    print(f"입력 경로: s3://{BUCKET_NAME}/meeting_logs/[file_id].json")
    print(f"출력 경로: s3://{BUCKET_NAME}/meeting_logs/[file_id]_final.json\n")
    
    for file_id in target_file_ids:
        # Step 1: 구조 분석
        meeting_data = analyze_structure(file_id)
        
        if meeting_data is None:
            print(f"⛔ {file_id}: 구조 분석 실패로 인해 상세 분석을 건너뜁니다.")
            continue
            
        # Step 2: 상세 분석 및 최종 저장
        success = analyze_details_and_consolidate(file_id, meeting_data)
        
        if not success:
            print(f"⛔ {file_id}: 상세 분석 실패")
        
        print("\n" + "="*80 + "\n")

