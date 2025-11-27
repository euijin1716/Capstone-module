import dataclasses
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

WhisperModels = Literal[
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]

@dataclass
class WhisperOptions:
    """Configuration options for WhisperSTT."""
    language: str
    model: WhisperModels | str
    device: str | None
    compute_type: str | None
    model_cache_directory: str | None
    warmup_audio: str | None


class WhisperSTT(stt.STT):
    """STT implementation using Whisper model."""

    def __init__(
            self,
            model: WhisperModels | str = "deepdml/faster-whisper-large-v3-turbo-ct2",
            language: str = "ko",
            device: Optional[str] = None,
            compute_type: Optional[str] = None,
            model_cache_directory: Optional[str] = None,
            warmup_audio: Optional[str] = None,
    ):
        """Initialize the WhisperSTT instance.

        Args:
            model: Whisper model to use
            language: Language code for speech recognition
            device: Device to use for inference (cuda, cpu, auto)
            compute_type: Compute type for inference (float16, int8, float32)
            model_cache_directory: Directory to store downloaded models
            warmup_audio: Path to audio file for model warmup
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._opts = WhisperOptions(
            language=language,
            model=model,
            device=device,
            compute_type=compute_type,
            model_cache_directory=model_cache_directory,
            warmup_audio=warmup_audio
        )

        self._model = None
        self._initialize_model()

        # Warmup the model with a sample audio if available
        if warmup_audio and os.path.exists(warmup_audio):
            self._warmup(warmup_audio)

    def _initialize_model(self):
        """Initialize the Whisper model."""
        device = self._opts.device
        compute_type = self._opts.compute_type

        logger.info(f"Using device: {device}, with compute: {compute_type}")

        # Ensure cache directories exist
        model_cache_dir = self._opts.model_cache_directory

        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            logger.info(f"Using model cache directory: {model_cache_dir}")

        self._model = WhisperModel(
            model_size_or_path=str(self._opts.model),
            device=device,
            compute_type=compute_type,
            download_root=model_cache_dir
        )
        logger.info("Whisper model loaded successfully")

    def _warmup(self, warmup_audio_path: str) -> None:
        """Performs a warmup transcription.

        Args:
            warmup_audio_path: Path to audio file for warmup
        """
        logger.info(f"Starting STT engine warmup using {warmup_audio_path}...")
        try:
            start_time = time.time()
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            segments, info = self._model.transcribe(warmup_audio_data,
                                                    language=self._opts.language,
                                                    beam_size=1)
            model_warmup_transcription = " ".join(segment.text for segment in segments)
            warmup_time = time.time() - start_time
            logger.info(f"STT engine warmed up in {warmup_time*1000:.1f}ms. Text: {model_warmup_transcription}")
        except Exception as e:
            logger.error(f"Failed to warm up STT engine: {e}")

    def update_options(
            self,
            *,
            model: Optional[WhisperModels | str] = None,
            language: Optional[str] = None,
            model_cache_directory: Optional[str] = None,
    ) -> None:
        """Update STT options.

        Args:
            model: Whisper model to use
            language: Language to detect
            model_cache_directory: Directory to store downloaded models
        """
        reinitialize = False

        if model:
            self._opts.model = model
            reinitialize = True

        if model_cache_directory:
            self._opts.model_cache_directory = model_cache_directory
            reinitialize = True

        if language:
            self._opts.language = language

        if reinitialize:
            self._initialize_model()

    def _sanitize_options(self, *, language: Optional[str] = None) -> WhisperOptions:
        """Create a copy of options with optional overrides.

        Args:
            language: Language override

        Returns:
            Copy of options with overrides applied
        """
        options = dataclasses.replace(self._opts)
        if language:
            options.language = language
        return options

    async def _recognize_impl(
            self,
            buffer: AudioBuffer,
            *,
            language: Optional[str],
            conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement speech recognition.

        Args:
            buffer: Audio buffer
            language: Language to detect
            conn_options: Connection options

        Returns:
            Speech recognition event
        """
        try:
            logger.info(f"Received audio, transcribing to text")
            options = self._sanitize_options(language=language)
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            # Convert WAV to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            start_time = time.time()
            segments, info = self._model.transcribe(
                audio_array,
                language=options.language,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=True,
                vad_filter=False, # LiveKit VAD를 이미 앞단에서 쓰므로 Whisper 내부 VAD는 끔

                # 1. 환청 억제: 말소리가 아닐 확률이 0.6 이상이면 빈 문자열 반환
                no_speech_threshold=0.6,

                # 2. 로그 확률 임계값: 모델의 확신이 -1.0보다 낮으면 무시 (너무 불확실한 단어 제거)
                log_prob_threshold=-1.0,

                #여기에 프롬프트를 넣으면 더 좋습니다.
                initial_prompt="이 문장은 회의 기록을 위한 한국어 대화입니다. 정확하게 기록해주세요" 
            )

            segments_list = list(segments)
            full_text = " ".join(segment.text.strip() for segment in segments_list)
            inference_time = time.time() - start_time

            logger.info(f"STT inference completed in {inference_time*1000:.1f}ms. Text: {full_text}")

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=options.language,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e