#%%
# 1. 라이브러리 임포트 및 API 키 설정
###############################################################################################################################################################################

import os
import google.generativeai as genai
import json
import argparse
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions
from dotenv import load_dotenv

# Import prompts from external file
import prompts

load_dotenv()

# Configuration
MODEL_NAME = "models/gemini-2.5-pro"
BUCKET_NAME = "hedj-s3-1"    # S3 버킷 이름

# S3 Client 초기화
s3_client = boto3.client('s3')

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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def generate_content_with_retry(model, prompt):
    """
    Gemini API 호출을 수행하며, 실패 시 지수 백오프로 재시도합니다.
    """
    return model.generate_content(prompt)

# ==============================================================================
# 2. Recap 생성 함수
# ==============================================================================
# ==============================================================================
# 2. Recap 생성 함수
# ==============================================================================
def generate_recap(file_id, end_utterance_id=None, input_folder="Request_Recap", output_folder="Recap"):
    print(f"\n{'='*80}")
    print(f"🚀 [Recap] 중간 요약 생성 시작: {file_id}")
    if end_utterance_id:
        print(f"   (Cut-off ID: {end_utterance_id})")
    print(f"{'='*80}\n")
    
    # S3 경로 설정
    input_s3_key = f"{input_folder}/{file_id}.json"
    
    try:
    # 1. S3에서 JSON 파일 읽기
        print(f"S3에서 파일 읽는 중: s3://{BUCKET_NAME}/{input_s3_key}")
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=input_s3_key)
        meeting_log_data = json.loads(response['Body'].read().decode('utf-8'))

        # 2. 대화 내용 추출 및 필터링
        utterances = meeting_log_data.get('utterances', [])
        participants = meeting_log_data.get('participants', [])
        speaker_map = {p['USER_ID']: p.get('name', f"P{i:02d}") for i, p in enumerate(participants)}

        conversation_text_lines = []

        found_cutoff = False
        for utterance in utterances:
            u_id = utterance.get('id')
            speaker_id = utterance.get('USER_ID')
            message = utterance.get('content')

            if speaker_id and message and u_id:
                speaker_label = speaker_map.get(speaker_id, speaker_id)
                conversation_text_lines.append(f"[ID: {u_id}] {speaker_label}: {message}")

            # end_utterance_id가 지정되어 있고, 현재 ID와 일치하면 중단
            if end_utterance_id and str(u_id) == str(end_utterance_id):
                found_cutoff = True
                break

        if end_utterance_id and not found_cutoff:
            print(f"⚠️ 경고: 지정된 Cut-off ID ({end_utterance_id})를 찾지 못했습니다. 전체 내용을 사용합니다.")

        conversation_text = "\n".join(conversation_text_lines)

        if not conversation_text:
            print("❌ 대화 내용이 없습니다.")
            return None

        print(f"✅ 분석 대상 발화 수: {len(conversation_text_lines)}개")

        # 3. 프롬프트 구성
        prompt_text = prompts.RECAP_PROMPT.format(input_data=conversation_text)

        # 4. Gemini API 호출
        model = genai.GenerativeModel(MODEL_NAME)
        print("--- Gemini API 호출 중 (Single-Shot) ---")

        response = generate_content_with_retry(model, prompt_text)

        # 5. 결과 파싱 및 출력
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        print(response)
        parsed_json = json.loads(json_string)

        print("\n" + "="*40)
        print("       📋 중간 요약 (Recap)       ")
        print("="*40)
        print(f"🔹 현재 주제: {parsed_json.get('current_topic', 'N/A')}")
        print("\n🔹 지금까지의 흐름:")
        for idx, item in enumerate(parsed_json.get('summary_so_far', [])):
            print(f"  {idx+1}. {item}")

        decisions = parsed_json.get('key_decisions', [])
        if decisions:
            print("\n🔹 주요 결정 사항:")
            for item in decisions:
                print(f"  - {item}")

        print(f"\n💡 Tip: {parsed_json.get('catch_up_tip', '')}")
        print("="*40 + "\n")

        # 6. 결과 S3 저장
        # 파일명 변환 로직: _request_recap -> _recap
        if file_id.endswith("_request_recap"):
            base_name = file_id.replace("_request_recap", "")
            output_filename = f"{base_name}_recap.json"
        else:
            output_filename = f"{file_id}_recap.json"

        if end_utterance_id:
            output_filename = output_filename.replace(".json", f"_{end_utterance_id}.json")

        output_s3_key = f"{output_folder}/{output_filename}"

        print(f"S3에 Recap 저장 중: s3://{BUCKET_NAME}/{output_s3_key}")
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=output_s3_key,
            Body=json.dumps(parsed_json, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
        print("✅ 저장 완료")

        return parsed_json

    except s3_client.exceptions.NoSuchKey:
        print(f"오류: S3에서 '{input_s3_key}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"Recap 생성 중 오류 발생: {e}")
        # traceback.print_exc() # 필요시 주석 해제
        return None

if __name__ == "__main__":
    # API 키 설정
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        exit(1)

    # Argument Parsing
    parser = argparse.ArgumentParser(description="늦게 온 참가자를 위한 회의 중간 요약 (Recap) 스크립트")
    parser.add_argument("--file_id", required=True, help="Target File ID (e.g., 'room001_20231121_143000')")
    parser.add_argument("--end_id", required=False, help="Optional: Cut-off Utterance ID (simulate 'current time')")
    parser.add_argument("--input_folder", default="Request_Recap", help="S3 Input Folder")
    parser.add_argument("--output_folder", default="Recap", help="S3 Output Folder")
    
    args = parser.parse_args()

    print("Recap 함수 시작")
    generate_recap(args.file_id, args.end_id, args.input_folder, args.output_folder)
