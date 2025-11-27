import os
import json
import asyncio
import boto3
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# AWS S3 설정
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

class S3Uploader:
    """
    S3 업로드를 전담하는 클래스
    """
    def __init__(self):
        self.s3_client = None
        if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET_NAME:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=AWS_REGION
                )
                print(f"☁️ S3 클라이언트 초기화 완료 (Bucket: {AWS_BUCKET_NAME})")
            except Exception as e:
                print(f"❌ S3 초기화 실패: {e}")
        else:
            print("⚠️ AWS 환경변수가 설정되지 않아 S3 업로드가 비활성화됩니다.")


    async def upload_json(self, data: dict, filename: str, folder: str = "meeting_logs"):
        """
        Python 딕셔너리 데이터를 JSON으로 변환하여 S3에 업로드
        """
        if not self.s3_client:
            return

        final_s3_key = f"{folder}/{filename}"
        print(f"⬆️ S3 업로드 시도... ({final_s3_key})")

        try:
            json_string = json.dumps(data, ensure_ascii=False, indent=4)

            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=AWS_BUCKET_NAME,
                Key=final_s3_key,
                Body=json_string,
                ContentType="application/json"
            )
            print("✅ S3 업로드 성공")
        except Exception as e:
            print(f"❌ S3 업로드 실패: {e}")

    async def read_json(self, key: str) -> dict | None:
        """
        S3에서 JSON 파일을 읽어 딕셔너리로 반환
        """
        if not self.s3_client:
            return None

        print(f"⬇️ S3 다운로드 시도... ({key})")
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=AWS_BUCKET_NAME,
                Key=key
            )
            content = await asyncio.to_thread(response['Body'].read)
            data = json.loads(content.decode('utf-8'))
            print("✅ S3 다운로드 성공")
            return data
        except Exception as e:
            print(f"❌ S3 다운로드 실패: {e}")
            return None
