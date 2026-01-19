"""
ONNX-based TTS Engine for Orator

Uses ONNX Runtime for model inference with pure NumPy.
Uses espeak-ng for G2P (grapheme-to-phoneme) conversion.
No PyTorch or kokoro dependency required.
"""

import os
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Generator

import numpy as np
import onnxruntime as ort

# Import pure G2P pipeline (no torch dependency)
from orator.g2p import PureG2PPipeline

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration"""
    model_path: str = "kokoro_model_onnx/kokoro-v1_0.onnx"
    tokenizer_path: str = "kokoro_model_onnx/tokenizer.json"
    voices_dir: str = "kokoro_model_onnx/voices"  # Use .bin voice files
    voice: str = "bf_isabella"
    hotkey_timeout: float = 0.5
    max_text_length: int = None
    device: str = "cpu"  # ONNX Runtime uses CPU by default
    speed: float = 1.0


@dataclass
class AudioData:
    """Audio data container"""
    samples: np.ndarray
    sample_rate: int
    duration: float


@dataclass
class AudioChunk:
    """Audio chunk data for streaming playback"""
    samples: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool


class ONNXTTSEngine:
    """TTS Engine using ONNX Runtime for inference (no PyTorch dependency)"""

    SAMPLE_RATE = 24000
    VOICE_PACK_SHAPE = (510, 256)  # Shape of voice pack: [max_phonemes, style_dim]

    def __init__(self, config: AppConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.vocab: dict = {}
        self.pipelines: dict = {}
        self.voice = config.voice
        self.voice_pack: Optional[np.ndarray] = None  # Pure numpy array
        self._model_loaded = False
        self._generation_lock = threading.Lock()

    def _get_available_providers(self) -> list:
        """Get available execution providers in priority order: CoreML > CUDA > CPU"""
        available_providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime available providers: {available_providers}")
        
        # Priority order: CoreML (MPS/Apple Silicon) > CUDA (GPU) > CPU
        priority_providers = []
        
        # Check for CoreMLExecutionProvider (Apple Silicon/MPS)
        if 'CoreMLExecutionProvider' in available_providers:
            priority_providers.append('CoreMLExecutionProvider')
            logger.info("CoreMLExecutionProvider (MPS) available - using for fast inference")
        
        # Check for CUDAExecutionProvider (NVIDIA GPU)
        if 'CUDAExecutionProvider' in available_providers:
            priority_providers.append('CUDAExecutionProvider')
            logger.info("CUDAExecutionProvider (GPU) available")
        
        # Always add CPU as fallback
        if 'CPUExecutionProvider' in available_providers:
            priority_providers.append('CPUExecutionProvider')
            logger.info("CPUExecutionProvider available as fallback")
        
        if not priority_providers:
            # Fallback to default if nothing found
            logger.warning("No execution providers found, using default")
            priority_providers = ['CPUExecutionProvider']
        
        return priority_providers

    def initialize_model(self) -> bool:
        """Load ONNX model and initialize pipelines"""
        try:
            logger.info("Initializing ONNX TTS model...")

            # Load tokenizer vocabulary
            if not os.path.exists(self.config.tokenizer_path):
                raise FileNotFoundError(
                    f"Tokenizer not found: {self.config.tokenizer_path}")

            with open(self.config.tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                self.vocab = tokenizer_data.get('model', {}).get('vocab', {})

            logger.info(f"Loaded vocabulary with {len(self.vocab)} tokens")

            # Load ONNX model
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(
                    f"ONNX model not found: {self.config.model_path}")

            logger.info(f"Loading ONNX model from: {self.config.model_path}")

            # Configure ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            # Determine best available execution provider
            # Priority: CoreML (MPS/Apple Silicon) > CUDA (GPU) > CPU
            providers = self._get_available_providers()
            
            logger.info(f"Using execution providers (in order): {providers}")
            
            # Create inference session with prioritized providers
            self.session = ort.InferenceSession(
                self.config.model_path,
                sess_options,
                providers=providers
            )

            logger.info("ONNX model loaded successfully")
            logger.info(f"Active provider: {self.session.get_providers()[0]}")
            logger.info(f"All available providers: {self.session.get_providers()}")

            # Initialize G2P pipeline for the configured voice's language
            voice_lang = self.voice[0]
            logger.info(f"Initializing G2P pipeline for language: {voice_lang}")

            try:
                # Create pure G2P pipeline (uses espeak-ng, no torch)
                self.pipelines[voice_lang] = PureG2PPipeline(lang_code=voice_lang)
                logger.info(f"G2P pipeline initialized for language: {voice_lang}")

            except Exception as e:
                logger.error(f"Failed to initialize G2P pipeline: {e}")
                raise RuntimeError(f"G2P pipeline initialization failed: {e}")

            # Validate voices directory
            if not os.path.exists(self.config.voices_dir):
                raise FileNotFoundError(
                    f"Voices directory not found: {self.config.voices_dir}")

            # Pre-load the default voice
            self._load_voice(self.voice)

            self._model_loaded = True

            # Pre-warm the model
            try:
                logger.info("Pre-warming ONNX model...")
                test_audio = self._generate_audio_internal("Hello")
                if test_audio is not None:
                    logger.info("Model pre-warming successful")
                else:
                    logger.warning("Model pre-warming failed, but continuing...")
            except Exception as e:
                logger.warning(f"Model pre-warming failed: {e}, but continuing...")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize ONNX TTS model: {e}")
            return False

    def _load_voice(self, voice_name: str) -> None:
        """Load a voice pack from .bin file (pure numpy, no torch)"""
        voice_file = os.path.join(self.config.voices_dir, f"{voice_name}.bin")

        if not os.path.exists(voice_file):
            raise FileNotFoundError(f"Voice file not found: {voice_file}")

        logger.info(f"Loading voice: {voice_name}")

        # Load voice pack as raw float32 numpy array and reshape
        # Voice packs are stored as [510, 256] float32 arrays in .bin format
        raw_data = np.fromfile(voice_file, dtype=np.float32)
        self.voice_pack = raw_data.reshape(self.VOICE_PACK_SHAPE)
        self.voice = voice_name

        # Update pipeline if language changed
        voice_lang = voice_name[0]
        if voice_lang not in self.pipelines:
            logger.info(f"Initializing G2P pipeline for new language: {voice_lang}")
            self.pipelines[voice_lang] = PureG2PPipeline(lang_code=voice_lang)

        logger.info(f"Voice {voice_name} loaded successfully")

    def set_voice(self, voice_name: str) -> bool:
        """Change the current voice"""
        try:
            self._load_voice(voice_name)
            return True
        except Exception as e:
            logger.error(f"Failed to set voice {voice_name}: {e}")
            return False

    def _phonemes_to_input_ids(self, phonemes: str) -> np.ndarray:
        """Convert phoneme string to input IDs using vocabulary"""
        input_ids = [0]  # Start token
        for p in phonemes:
            if p in self.vocab:
                input_ids.append(self.vocab[p])
        input_ids.append(0)  # End token
        return np.array([input_ids], dtype=np.int64)

    def _get_style_vector(self, phoneme_length: int) -> np.ndarray:
        """Get style vector from voice pack for given phoneme length.
        
        Returns shape [1, 256] for ONNX model input.
        """
        # Index into voice pack and add batch dimension
        # voice_pack has shape [510, 256], we need [1, 256]
        idx = min(phoneme_length - 1, self.VOICE_PACK_SHAPE[0] - 1)
        style = self.voice_pack[idx]  # Shape: [256]
        return style.reshape(1, -1)  # Shape: [1, 256]

    def _run_inference(self, input_ids: np.ndarray, ref_s: np.ndarray, speed: float) -> np.ndarray:
        """Run ONNX inference"""
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")

        # Get input names from model
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]

        # Prepare inputs - ensure correct dtypes
        inputs = {
            input_names[0]: input_ids.astype(np.int64),
            input_names[1]: ref_s.astype(np.float32),
            input_names[2]: np.array([speed], dtype=np.float32)
        }

        # Run inference
        outputs = self.session.run(output_names, inputs)

        # First output is the audio waveform
        return outputs[0]

    def generate_audio(self, text: str) -> Optional[AudioData]:
        """Convert text to audio with thread safety"""
        with self._generation_lock:
            return self._generate_audio_with_splitting(text)

    def generate_audio_stream(self, text: str, chunk_size: int = 200) -> Generator[AudioChunk, None, None]:
        """Generate audio chunks as a stream for immediate playback"""
        with self._generation_lock:
            yield from self._generate_audio_stream_internal(text, chunk_size)

    def _generate_audio_with_splitting(self, text: str) -> Optional[AudioData]:
        """Generate audio for text by processing through G2P and ONNX"""
        try:
            if not self._model_loaded or self.session is None:
                logger.error("Model not initialized")
                return None

            if not text or not text.strip():
                logger.warning("Empty text provided")
                return None

            text = text.strip()
            logger.info(f"Generating audio for text: {len(text)} characters")

            voice_lang = self.voice[0]
            pipeline = self.pipelines[voice_lang]

            audio_segments = []
            total_duration = 0

            # Process text through G2P pipeline
            for segment_idx, (graphemes, phonemes, _) in enumerate(
                pipeline(text, self.voice, speed=self.config.speed)
            ):
                if not phonemes:
                    continue

                try:
                    # Convert phonemes to input IDs
                    input_ids = self._phonemes_to_input_ids(phonemes)

                    # Get reference style from voice pack (pure numpy)
                    ref_s = self._get_style_vector(len(phonemes))

                    # Run ONNX inference
                    audio = self._run_inference(input_ids, ref_s, self.config.speed)

                    # Squeeze if needed
                    if audio.ndim > 1:
                        audio = audio.squeeze()

                    audio_segments.append(audio)
                    segment_duration = len(audio) / self.SAMPLE_RATE
                    total_duration += segment_duration
                    logger.info(f"Generated segment {segment_idx + 1}: {segment_duration:.2f}s")

                except Exception as e:
                    logger.error(f"Failed to generate segment {segment_idx}: {e}")
                    continue

            if not audio_segments:
                logger.error("No audio segments generated")
                return None

            # Concatenate all segments
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                final_audio = np.concatenate(audio_segments)

            return AudioData(
                samples=final_audio,
                sample_rate=self.SAMPLE_RATE,
                duration=total_duration
            )

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    def _generate_audio_internal(self, text: str) -> Optional[AudioData]:
        """Internal audio generation for single segment (used for testing)"""
        try:
            if not self._model_loaded or self.session is None:
                logger.error("Model not initialized")
                return None

            if not text or not text.strip():
                return None

            text = text.strip()
            voice_lang = self.voice[0]
            pipeline = self.pipelines[voice_lang]

            # Process just the first segment
            for graphemes, phonemes, _ in pipeline(text, self.voice, speed=self.config.speed):
                if not phonemes:
                    continue

                input_ids = self._phonemes_to_input_ids(phonemes)
                ref_s = self._get_style_vector(len(phonemes))
                audio = self._run_inference(input_ids, ref_s, self.config.speed)

                if audio.ndim > 1:
                    audio = audio.squeeze()

                return AudioData(
                    samples=audio,
                    sample_rate=self.SAMPLE_RATE,
                    duration=len(audio) / self.SAMPLE_RATE
                )

            return None

        except Exception as e:
            logger.error(f"Internal audio generation failed: {e}")
            return None

    def _split_text_simple(self, text: str, chunk_size: int = 200) -> list:
        """Split text into chunks at sentence boundaries"""
        import re

        if not text or len(text) <= chunk_size:
            return [text] if text else []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        if not chunks and text:
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        return chunks

    def _generate_audio_stream_internal(self, text: str, chunk_size: int = 200) -> Generator[AudioChunk, None, None]:
        """Generate audio chunks as a stream"""
        try:
            if not self._model_loaded or self.session is None:
                logger.error("Model not initialized")
                return

            if not text or not text.strip():
                return

            text = text.strip()
            logger.info(f"Generating streaming audio for: {len(text)} characters")

            text_chunks = self._split_text_simple(text, chunk_size)
            if not text_chunks:
                return

            voice_lang = self.voice[0]
            pipeline = self.pipelines[voice_lang]

            for chunk_index, text_chunk in enumerate(text_chunks):
                try:
                    logger.info(f"Processing chunk {chunk_index + 1}/{len(text_chunks)}: \"{text_chunk}\"")

                    chunk_audio_segments = []

                    for segment_idx, (graphemes, phonemes, _) in enumerate(
                        pipeline(text_chunk, self.voice, speed=self.config.speed)
                    ):
                        if not phonemes:
                            continue

                        try:
                            input_ids = self._phonemes_to_input_ids(phonemes)
                            ref_s = self._get_style_vector(len(phonemes))
                            audio = self._run_inference(input_ids, ref_s, self.config.speed)

                            if audio.ndim > 1:
                                audio = audio.squeeze()

                            chunk_audio_segments.append(audio)

                        except Exception as e:
                            logger.error(f"Failed segment {segment_idx} in chunk {chunk_index}: {e}")
                            continue

                    if chunk_audio_segments:
                        if len(chunk_audio_segments) == 1:
                            chunk_audio = chunk_audio_segments[0]
                        else:
                            chunk_audio = np.concatenate(chunk_audio_segments)

                        yield AudioChunk(
                            samples=chunk_audio,
                            sample_rate=self.SAMPLE_RATE,
                            chunk_index=chunk_index,
                            is_final=(chunk_index == len(text_chunks) - 1)
                        )

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    continue

            logger.info("Streaming audio generation completed")

        except Exception as e:
            logger.error(f"Streaming audio generation failed: {e}")

    def cleanup(self) -> None:
        """Free resources"""
        try:
            self.session = None
            self.voice_pack = None
            self.pipelines.clear()
            logger.info("ONNX TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded and self.session is not None
