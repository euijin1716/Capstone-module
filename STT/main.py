import asyncio
import os
import datetime
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import requests
import sys

#최종 summarize할거면 1 끌거면 0
Summarize_enable = 1

# [AI & ML 라이브러리]
import google.generativeai as genai
from transformers import pipeline
import torch

# [LiveKit 라이브러리]
from livekit import rtc, agents
from livekit.agents import JobContext, WorkerOptions, cli, stt
from livekit.plugins import silero

# [로컬 플러그인] WhisperSTT 클래스가 정의된 파일
from whisper_plugin import WhisperSTT
from logger import TranscriptLogger

# .env 파일 로드
load_dotenv()

# --- 환경 변수 설정 ---
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")



# Google Gemini API 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️ [경고] GOOGLE_API_KEY가 설정되지 않았습니다. 투표 기능이 작동하지 않습니다.")




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

class VoteManager:
    """
    [단순화 + 한국어 레이블 버전] 한 문장 단위로 투표/안건 제안 발화를 감지하는 매니저

    동작 순서:
    1) STT 한 문장이 들어오면 zero-shot 분류기로 '결정을 요청하는 발화'인지 판단
    2) 아니면 종료
    3) 맞으면 최근 발화(최대 25줄)를 컨텍스트로 포함해 Gemini에 전달
    4) Gemini가 투표라고 판단하면, 현재 문장을 기준으로 주제/선택지를 추출
    5) 추출된 데이터로 VOTE_CREATED 이벤트를 LiveKit data channel로 전송
    """
    def __init__(self, room: rtc.Room):
        self.room = room

        # Gemini 2.0 Flash 초기화 (JSON 모드)
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0 
            }
        )

        # 최근 발화 저장용 슬라이딩 윈도우 버퍼 (최대 25줄)
        self.transcript_buffer: list[str] = []
        self.max_buffer_size = 25

        # 제로샷 분류 모델 (한 문장 단위 사용)
        device = "cuda" if torch.cuda.is_available() else -1
        print(f"🧠 [VoteManager] 한국어 레이블 기반 감지 모드 (device={device})")

        # ✅ 한국어 문장형 레이블 + hypothesis_template 설정
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=device,
            hypothesis_template="이 문장은 {}."
        )

        # ✅ 우리가 진짜 잡고 싶은 positive 레이블
        self.POS_LABEL = "회의 참여자들에게 제안하거나 의견을 물어보는 투표가 필요한 발화"

        # ✅ 부정 레이블들 (인사 / 잡담 / 설명 등)
        self.candidate_labels = [
            self.POS_LABEL,
            "단순히 인사나 안부를 전하는 발화",
            "가벼운 잡담이나 농담처럼 아무것도 결정하지 않는 발화",
            "정보를 전달하거나 상황을 설명할 뿐, 결정을 요구하지 않는 발화",
            "회의 진행을 위한 발화",
            "회의 참여자들에게 의견을 물어보지만 투표가 필요하지 않은 발화"
        ]

        # 중복 방지용
        self.last_vote_topic: str | None = None
        self.last_vote_time: float = 0.0
        self.cooldown_sec = 30  # 같은 주제 연속 방지용 (원하면 조절 가능)

    def add_transcript(self, participant_name: str, text: str):
        """
        STT에서 최종 문장이 들어올 때마다 호출됨.
        - 버퍼에 추가
        - 해당 문장을 대상으로 zero-shot 분류 + Gemini 분석 태스크 실행
        """
        line = f"{participant_name}: {text}"
        self.transcript_buffer.append(line)
        if len(self.transcript_buffer) > self.max_buffer_size:
            self.transcript_buffer.pop(0)

        asyncio.create_task(self._handle_utterance(participant_name, text))

    async def _handle_utterance(self, participant_name: str, text: str):
        """
        1) 제로샷 분류로 이 문장이 '결정/선택 요청 발화'인지 판단
        2) 맞으면 Gemini에 컨텍스트 포함 분석 요청
        """
        now = asyncio.get_event_loop().time()
        if now - self.last_vote_time < self.cooldown_sec:
            # 너무 짧은 시간 안에 여러 번 뜨는 것 방지 (원하면 제거 가능)
            return

        # 1) zero-shot 분류 (한 문장만)
        try:
            zs_result = await asyncio.to_thread(
                self.classifier,
                text,
                self.candidate_labels,
                multi_label=False,
            )
        except Exception as e:
            print(f"❌ [VoteManager/ZSL] 제로샷 분류 에러: {e}")
            return

        top_label = zs_result["labels"][0]
        top_score = zs_result["scores"][0]
        print(f"🔎 [Zero-shot] \"{text}\" -> {top_label} ({top_score:.2f})")

        print(f"\n📊 [제로샷 분류 결과] (최근 1문장 기준)") # 로그도 수정
        for l, s in zip(zs_result['labels'], zs_result['scores']):
            print(f"   - {l}: {s:.4f}")
        print("-" * 30)

        # ✅ 투표/결정 요청 발화로 볼 기준 (threshold는 나중에 튜닝)
        is_decision_like = (top_label == self.POS_LABEL)# and top_score >= 0.5)

        if not is_decision_like:
            print("투표 제안이 아니라고 판단함.")
            return

        # 2) Gemini 분석 (컨텍스트 + 현재 문장)
        await self._analyze_with_gemini(participant_name, text)

    async def _analyze_with_gemini(self, participant_name: str, text: str):
        """
        Gemini에 최근 대화(최대 25줄)와 현재 문장을 넘겨:
        - 이 발언이 실제 투표 제안인지 최종 판단
        - 투표라면 '주제'와 '선택지'를 현재 문장 기준으로 추출
        """
        context_text = "\n".join(self.transcript_buffer)

        system_prompt = (
            "당신은 회의 대화를 분석하는 AI 서기이다.\n"
            "아래 대화의 흐름을 참고하되, 마지막에 주어진 [후보 문장]이 실제로 "
            "'투표를 제안하거나 의견을 모으기 위한 발언'인지 판단해야 한다.\n"
            "\n"
            "[판단 규칙]\n"
            "1. 후보 문장이 다음과 같은 의미를 가지면 투표 제안으로 간주한다.\n"
            "   - '무엇으로 할지 정하자', '투표하자', '어떤 걸로 할까요?', "
            "     '1번/2번 중에 골라주세요' 등 구체적인 선택을 요청하는 경우.\n"
            "   - 일정/장소/방식 등 여러 옵션 중 하나를 고르게 하는 경우.\n"
            "   - 회의 안건에 대해 '찬성/반대' 의견을 묻는 경우.\n"
            "2. 단순 제안, 정보 설명, 농담, 잡담만 하는 경우는 투표로 보지 않는다.\n"
            "3. 주제(topic)는 '점심 메뉴 선정', '다음 회의 일정 결정' 처럼 "
            "   짧은 명사형으로 요약한다.\n"
            "4. 선택지(options)는 후보 문장이나 바로 인접한 발화에 명시된 것만 사용하고, "
            "   없으면 빈 배열([])로 둔다.\n"
            "\n"
            "[출력 형식]\n"
            "반드시 JSON 한 개만 반환하라.\n"
            "1) 투표가 필요한 경우:\n"
            "{\n"
            "  \"is_vote\": true,\n"
            "  \"topic\": \"짧고 명사형의 주제\",\n"
            "  \"options\": [\"옵션1\", \"옵션2\"]\n"
            "}\n"
            "2) 투표가 필요하지 않은 경우:\n"
            "{ \"is_vote\": false }\n"
        )

        prompt = (
            f"{system_prompt}\n\n"
            "[대화 전체 컨텍스트]\n"
            f"{context_text}\n\n"
            "[후보 문장]\n"
            f"{participant_name}: {text}\n"
        )

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            print("Gemini 호출 시작")
        except Exception as e:
            print(f"❌ [VoteManager/Gemini] 호출 에러: {e}")
            return

        result_text = response.text
        try:
            result_json = json.loads(result_text)
        except Exception:
            print(f"⚠️ [VoteManager/Gemini] JSON 파싱 실패, 원문: {result_text[:200]}...")
            return

        if isinstance(result_json, list):
            result_json = result_json[0] if result_json else {}

        if not isinstance(result_json, dict):
            return

        if not result_json.get("is_vote"):
            print("ℹ️ [VoteManager] Gemini: 투표 아님으로 판단")
            return

        topic = result_json.get("topic")
        options = result_json.get("options", [])

        if not topic:
            print("⚠️ [VoteManager] topic이 비어 있어 투표 생성 중단")
            return

        if not isinstance(options, list):
            options = []

        print(f"✨ [투표 감지] topic={topic}, options={options}, proposer={participant_name}")

        self.last_vote_topic = topic
        self.last_vote_time = asyncio.get_event_loop().time()

        vote_payload = {
            "type": "VOTE_CREATED",
            "data": {
                "topic": topic,
                "options": options,  # 없으면 [] 전달
                "proposer": participant_name,
                "created_at": datetime.datetime.now().isoformat(),
            },
        }

        try:
            await self.room.local_participant.publish_data(
                payload=json.dumps(vote_payload, ensure_ascii=False).encode("utf-8"),
                reliable=True,
            )
            print("📨 [VoteManager] VOTE_CREATED 이벤트 전송 완료")
        except Exception as e:
            print(f"❌ [VoteManager] LiveKit publish_data 에러: {e}")







async def process_track(participant: rtc.RemoteParticipant, track: rtc.RemoteAudioTrack, stt_provider, vad_provider, logger, vote_manager):
    """
    오디오 트랙 처리 파이프라인
    1. Resampling (48k -> 16k)
    2. STT (Whisper)
    3. Logging & Voting Analysis
    """
    print(f"[{participant.identity}] 오디오 트랙 처리 시작")

    audio_stream = rtc.AudioStream(track)
    resampler = rtc.AudioResampler(input_rate=48000, output_rate=16000)

    stream_adapter = stt.StreamAdapter(stt=stt_provider, vad=vad_provider)
    stt_stream = stream_adapter.stream()

    async def feed_audio():
        try:
            async for event in audio_stream:
                resampled_frames = resampler.push(event.frame)
                for frame in resampled_frames:
                    stt_stream.push_frame(frame)
        except Exception as e:
            print(f"[{participant.identity}] 오디오 입력 중단: {e}")
        finally:
            stt_stream.end_input()

    asyncio.create_task(feed_audio())

    try:
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = event.alternatives[0].text.strip()
                if text:
                    print(f"🗣️ [{participant.identity}]: {text}")
                    # 1. 로그 저장
                    logger.log(participant.identity, text)
                    # 2. 투표 매니저에게 전달 (여기서 분석 로직 시작)
                    vote_manager.add_transcript(participant.identity, text)

            elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                pass

    except Exception as e:
        print(f"[{participant.identity}] STT 처리 에러: {e}")
    finally:
        await stt_stream.aclose()

async def periodic_upload_task(logger, interval=300):
    try:
        while True:
            await asyncio.sleep(interval)
            print(f"⏰ 정기 백업 수행 ({interval}초)")
            await logger.upload_to_s3()
    except asyncio.CancelledError:
        pass

async def entrypoint(ctx: JobContext):
    print("Job 시작. 초기화 중...")
    transcript_logger = TranscriptLogger(ctx.room)
    vote_manager = VoteManager(ctx.room)
    upload_task = None

    try:
        print("Whisper 모델 로딩 중...")
        stt_instance = await asyncio.to_thread(
            WhisperSTT,
            model="deepdml/faster-whisper-large-v3-turbo-ct2",
            language="ko",
            device="cuda",
            compute_type="float16"
        )
        print("Silero VAD 모델 로드 중...")
        # [수정] 작은 소리 감지를 위해 0.1초로 민감도 상향
        vad_instance = await asyncio.to_thread(
            silero.VAD.load,
            min_speech_duration=0.1,
            min_silence_duration=2.0,
        )

        await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
        print(f"방 접속 완료: {ctx.room.name}")

        upload_task = asyncio.create_task(periodic_upload_task(transcript_logger, interval=300))

        # [추가] 초기 접속자 등록
        for p in ctx.room.remote_participants.values():
            transcript_logger.add_participant(p)

        @ctx.room.on("participant_connected")
        def on_participant_connected(participant):
            print(f"👋 참가자 입장: {participant.identity}")
            transcript_logger.add_participant(participant)

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(process_track(participant, track, stt_instance, vad_instance, transcript_logger, vote_manager))

        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            print(f"👋 참가자 퇴장: {participant.identity}")
            if len(ctx.room.remote_participants) == 0:
                print("🚪 모든 참가자 퇴장 -> 종료 프로세스 시작")
                
                async def shutdown_sequence():
                    # 1. Upload raw logs
                    await transcript_logger.upload_to_s3()
                    
                    if Summarize_enable == 1:
                        print("📝 [Summarize] 요약 프로세스 시작")
                        room_name = ctx.room.name
                        
                        # 2. Status -> IN_PROGRESS
                        update_session_status(room_name, "IN_PROGRESS")
                        
                        # 3. Run Summarization
                        base_name = os.path.basename(transcript_logger.filename).replace('.jsonl', '')
                        script_path = os.path.join("../Summarize", "S3_Summarization.py")
                        script_path = os.path.abspath(script_path)
                        
                        command = [
                            r"C:\Users\salus\IdeaProjects\untitled1\.venv\Scripts\python.exe", script_path,
                            "--file_ids", base_name
                        ]
                        
                        print(f"🚀 S3_Summarization.py 실행: {' '.join(command)}")
                        
                        try:
                            process = await asyncio.create_subprocess_exec(
                                *command,
                                #stdout=asyncio.subprocess.PIPE,
                                #stderr=asyncio.subprocess.PIPE
                            )
                            
                            stdout, stderr = await process.communicate()
                            
                            # if stdout:
                            #     print(f"[S3_Summarization Output]\n{stdout.decode()}")
                            # if stderr:
                            #     print(f"[S3_Summarization Error]\n{stderr.decode()}")

                            rc = await process.wait()

                            if rc == 1:
                                print("✅ 요약 완료")
                                # 4. Status -> COMPLETED
                                update_session_status(room_name, "COMPLETED")
                            else:
                                print(f"❌ 요약 스크립트 실패 (Exit Code: {process.returncode})")
                                
                        except Exception as e:
                            print(f"❌ 요약 프로세스 실행 중 에러: {e}")
                    
                    print("🛑 Agent 종료")
                    ctx.shutdown()

                asyncio.create_task(shutdown_sequence())

        @ctx.room.on("data_received")
        def on_data_received(data_packet: rtc.DataPacket):
            """데이터 채널 메시지 수신 처리"""
            try:
                decoded_str = data_packet.data.decode("utf-8")
                message = json.loads(decoded_str)
                print(f"📨 데이터 수신: {message} from {data_packet.participant.identity}")

                if message.get("action") == "Request_Recap":
                    print("📢 [Request_Recap] 요청 수신 -> S3 업로드 시작")
                    requester_id = data_packet.participant.identity
                    
                    async def handle_recap_request(target_id):
                        # 1. S3 업로드 (await로 완료 대기)
                        # 파일명: {base_name}_request_recap.json
                        # 여기서 base_name을 알기 위해 logger의 filename을 참조하거나,
                        # upload_to_s3가 업로드한 파일명을 리턴하게 하면 좋겠지만,
                        # 현재 구조상 logger.filename에서 유추 가능.
                        
                        # logger.filename 예: logs/roomname_timestamp.jsonl
                        # upload_to_s3 호출 시 suffix="_request_recap" -> 업로드 파일명: roomname_timestamp_request_recap.json
                        
                        base_name = os.path.basename(transcript_logger.filename).replace('.jsonl', '')
                        file_id = f"{base_name}_request_recap" # .json 제외
                        
                        await transcript_logger.upload_to_s3(
                            folder="Request_Recap", 
                            suffix="_request_recap"
                        )
                        
                        print("✅ S3 업로드 완료 -> S3_Recap.py 실행")
                        
                        # 2. S3_Recap.py 실행
                        # python Summarize/S3_Recap.py --file_ids {file_id} --input_folder Request_Recap --output_folder Recap
                        
                        # 현재 작업 디렉토리 기준 상대 경로
                        script_path = os.path.join("../Summarize", "S3_Recap.py")

                        command = [
                            r"C:\Users\salus\IdeaProjects\untitled1\.venv\Scripts\python.exe", script_path,
                            "--file_id", file_id,
                            "--input_folder", "Request_Recap",
                            "--output_folder", "Recap"
                        ]
                        
                        try:
                            # 비동기로 서브프로세스 실행 (결과 기다리지 않음 or 기다림 선택)
                            # 여기서는 실행만 시켜두고 로그만 확인
                            process = await asyncio.create_subprocess_exec(
                                *command#,
                                #stdout=asyncio.subprocess.PIPE,
                                #stderr=asyncio.subprocess.PIPE
                            )
                            print(f"🚀 S3_Recap.py 실행됨 (PID: {process.pid})")
                            
                            # (선택) 출력을 실시간으로 보거나 나중에 확인
                            #stdout, stderr = await process.communicate()
                            #if stdout: print(f"[S3_Recap] {stdout.decode()}")
                            #if stderr: print(f"[S3_Recap Error] {stderr.decode()}")
                            
                            # 3. 결과 S3에서 읽어오기
                            # 예상되는 파일명: Recap/{base_name}_recap.json
                            recap_key = f"Recap/{base_name}_recap.json"

                            recap_data = await fetch_recap_with_retry(transcript_logger, recap_key)
                            if recap_data is None:
                            # 여기서 포기 처리 / 로그 / 예외 등 원하는 대로
                                print("❌ Recap 생성 실패(시간 초과)")
                                return
                            
                            if recap_data:
                                print(f"✅ Recap 데이터 S3 로드 성공 -> LiveKit 전송 (Target: {target_id})")
                                
                                payload = {
                                    "type": "RECAP_GENERATED",
                                    "data": recap_data
                                }
                                
                                await ctx.room.local_participant.publish_data(
                                    payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                    reliable=True,
                                    destination_identities=[target_id]
                                )
                                print("📨 [RECAP_GENERATED] 이벤트 전송 완료")
                            else:
                                print("❌ Recap 데이터 로드 실패")
                            
                        except Exception as e:
                            print(f"❌ S3_Recap.py 실행 실패: {e}")

                    asyncio.create_task(handle_recap_request(requester_id))

            except Exception as e:
                print(f"❌ 데이터 수신 처리 중 에러: {e}")

        for p in ctx.room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.create_task(process_track(p, pub.track, stt_instance, vad_instance, transcript_logger, vote_manager))

        await asyncio.Event().wait()

    except Exception as e:
        print(f"❌ 메인 루프 에러 발생: {e}")
    finally:
        print("작업 종료 처리 중...")
        if upload_task: upload_task.cancel()
        await transcript_logger.upload_to_s3()
        ctx.shutdown()



async def fetch_recap_with_retry(transcript_logger, recap_key: str,
                                 max_retries: int = 10,
                                 delay_seconds: int = 30):
    """
    S3에서 recap JSON을 읽되, 파일이 아직 없으면 기다렸다가 재시도함.
    - max_retries: 최대 재시도 횟수
    - delay_seconds: 각 시도 사이 대기 시간(초)
    """
    for attempt in range(1, max_retries + 1):
        try:
            recap_data = await transcript_logger.s3_uploader.read_json(recap_key)
            if recap_data is not None:
                print(f"✅ Recap found on attempt {attempt}")
                return recap_data
            # read_json이 '없으면 None'을 리턴하는 형태라면 여기로 떨어짐
            print(f"⏳ Recap not ready yet (attempt {attempt}/{max_retries}), retrying in {delay_seconds}s...")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "NoSuchKey":
                # S3에 아직 파일이 없을 때
                print(f"⏳ Recap object not found (attempt {attempt}/{max_retries}), retrying in {delay_seconds}s...")
            else:
                # 다른 S3 에러면 바로 터뜨림
                raise

        # 여기까지 왔으면 아직 파일이 없는 상황 → 잠깐 대기
        await asyncio.sleep(delay_seconds)

    print("⚠️ Recap still not available after all retries.")
    return None

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))