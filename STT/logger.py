import os
import json
import datetime
from livekit import rtc
from S3_upload import S3Uploader

class TranscriptLogger:
    """
    STT 결과를 로컬 JSONL 파일로 저장하고 주기적으로 S3에 업로드하는 로거
    - 절대 경로 사용으로 파일 저장 위치 보장
    - Append 방식으로 로컬 저장, Overwrite 방식으로 S3 업로드
    - S3 업로드 시 메타데이터와 참여자 정보를 포함한 확장된 JSON 포맷 사용
    """
    def __init__(self, room: rtc.Room):
        self.room = room
        self.room_name = room.name
        # [수정] 절대 경로 사용하여 파일 저장 위치 명확화
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = datetime.datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.log_dir, f"{self.room_name}_{timestamp}.jsonl")
        self.utterance_id = 1
        
        # S3 업로더 초기화
        self.s3_uploader = S3Uploader()

        # [추가] 참여자 이력 관리 (퇴장한 사람도 포함하기 위함)
        # Key: identity, Value: Participant Data Dict
        self.participants_history = {}

    def add_participant(self, participant: rtc.RemoteParticipant):
        """참여자 입장 시 정보 저장"""
        meta = {}
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
            except:
                pass
        
        p_data = {
            "USER_ID": participant.identity,
            "name": participant.name if participant.name else participant.identity,
            "age": meta.get("age", "unknown"),
            "occupation": meta.get("occupation", "unknown"),
            "role": meta.get("role", "unknown"),
            "sex": meta.get("sex", "unknown")
        }
        
        # 이미 있으면 업데이트, 없으면 추가
        self.participants_history[participant.identity] = p_data
        print(f"📝 [Logger] 참여자 기록 추가: {participant.identity}")

    def log(self, participant_id, text):
        """개별 발화 내용을 로컬 파일에 기록"""
        now = datetime.datetime.now().isoformat()
        entry = {
            "id": self.utterance_id,
            "start_time": now,
            "USER_ID": participant_id, # 요청에 따라 USER_ID로 변경
            "content": text
        }
        self.utterance_id += 1
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _get_metadata(self):
        """방 메타데이터 생성"""
        # 실제 발화자 수는 로그 파일을 읽어서 계산해야 정확하지만, 
        # 여기서는 현재 방에 있는 사람 수 등으로 근사하거나 나중에 계산할 수 있음.
        # 일단 전체 참여자 수와 동일하게 처리하거나 별도 로직 필요.
        # 여기서는 단순화를 위해 현재 접속자 수 사용.
        
        return {
            "roomname": self.room_name,
            "date": self.start_time.isoformat(),
            "participant_num": len(self.participants_history) + 1, # 전체 누적 참여자 수 (로컬 포함)
            "speaker_num": 0 # 업로드 시점에 계산
        }

    def _get_participants_data(self):
        """참여자 정보 수집 (이력 기준)"""
        # 저장된 모든 참여자 이력 반환
        return list(self.participants_history.values())

    async def upload_to_s3(self, folder: str = "meeting_logs", suffix: str = ""):
        """로컬 파일을 읽어 확장된 JSON 형태로 변환 후 S3에 업로드"""
        if not os.path.exists(self.filename):
            return

        utterances_list = []
        speaker_set = set()
        
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        utterances_list.append(data)
                        speaker_set.add(data.get("USER_ID"))
        except Exception as e:
            print(f"❌ 로그 파일 읽기 실패: {e}")
            return

        # 메타데이터 구성
        metadata = self._get_metadata()
        metadata["speaker_num"] = len(speaker_set)
        # participant_num 업데이트 (로그에 기록된 모든 사람 포함 or 현재 접속자)
        # 여기서는 현재 접속자 기준으로 하되, 로그에 있는 사람이 나갔을 수도 있으니
        # 로그에 있는 사람 + 현재 접속자 합집합으로 하는게 더 정확할 수 있음.
        # 일단 요청된 포맷에 맞춤.
        
        final_json_data = {
            "metadata": metadata,
            "participants": self._get_participants_data(),
            "utterances": utterances_list
        }
        
        # 파일명 생성 (.jsonl -> .json)
        # suffix가 있으면 추가 (예: _request_recap)
        base_name = os.path.basename(self.filename).replace('.jsonl', '')
        json_filename = f"{base_name}{suffix}.json"
        
        await self.s3_uploader.upload_json(final_json_data, json_filename, folder=folder)
