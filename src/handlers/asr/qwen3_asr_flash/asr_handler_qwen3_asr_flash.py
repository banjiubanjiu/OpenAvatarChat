import re
import json
import time
import base64
import requests
import tempfile
import wave
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from abc import ABC
import os
from queue import Queue, Empty
import asyncio
from concurrent.futures import ThreadPoolExecutor
try:
    import dashscope
except ImportError:
    logger.warning("dashscope not installed. Please install with: pip install dashscope")
    dashscope = None

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext
from engine_utils.directory_info import DirectoryInfo


class Qwen3ASRConfig(HandlerBaseConfigModel, BaseModel):
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    model_name: str = Field(default="qwen3-asr-flash")
    base_url: str = Field(default="https://dashscope.aliyuncs.com/api/v1")
    sample_rate: int = Field(default=16000)
    language: str = Field(default="zh")
    enable_itn: bool = Field(default=True)  # 数字转文字
    stream: bool = Field(default=False)  # 是否流式输出


class Qwen3ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.audio_buffer = []
        self.speech_id = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.shared_states = None
        self.transcription_result = ""

        # 音频调试功能
        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), "dump_qwen3_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")


class HandlerQwen3ASR(HandlerBase):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="Qwen3-ASR-Flash",
            config_model=Qwen3ASRConfig
        )

    def initialize(self, config: HandlerBaseConfigModel):
        self.config = cast(Qwen3ASRConfig, config)
        logger.info(f"Initializing Qwen3-ASR with model: {self.config.model_name}")

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(Qwen3ASRContext, context)

        logger.info(f"Qwen3-ASR handle called with input type: {inputs.type}")

        if inputs.type != ChatDataType.HUMAN_AUDIO:
            logger.debug(f"Skipping non-audio input: {inputs.type}")
            return

        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is not None:
            context.speech_id = speech_id

        audio = inputs.data.get_main_data()
        if audio is None:
            logger.warning("Received audio input but no main data")
            return

        audio = audio.squeeze()
        logger.info(f'audio in, shape={audio.shape}, speech_id={context.speech_id}')

        # Accumulate audio data
        if audio.shape[0] > 0:
            context.audio_buffer.append(audio)

        # Check if speech ended - then process the accumulated audio
        speech_end = inputs.data.get_meta("human_speech_end", False)
        if speech_end:
            logger.info(f"Speech ended for session {context.session_id}, processing accumulated audio")

            # Process accumulated audio synchronously and yield results
            if context.audio_buffer:
                try:
                    # Concatenate all audio chunks
                    full_audio = np.concatenate(context.audio_buffer)

                    if len(full_audio) > 0:
                        logger.info(f"Processing {len(full_audio)} samples ({len(full_audio)/16000:.2f}s)")

                        # Create temporary WAV file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_path = temp_file.name

                            # VAD outputs float32 samples in [-1, 1]; convert to int16 PCM for wav
                            if full_audio.dtype != np.int16:
                                normalized_audio = np.clip(full_audio, -1.0, 1.0)
                                pcm_audio = (normalized_audio * 32767).astype(np.int16)
                            else:
                                pcm_audio = full_audio

                            # Write WAV file
                            with wave.open(temp_path, 'wb') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(context.config.sample_rate)
                                wav_file.writeframes(pcm_audio.tobytes())

                        logger.info(f"Created temporary audio file: {temp_path}")

                        # Call Qwen3-ASR API
                        transcript = ""
                        if context.config.stream:
                            logger.info("Using streaming mode for Qwen3-ASR")
                            yield from self._stream_qwen3_asr_api(context, temp_path, output_definition)
                        else:
                            transcript = self._call_qwen3_asr_api(context, temp_path)
                            logger.info(f"Transcription result: {transcript}")

                        # Clean up temp file
                        os.unlink(temp_path)

                        # 调试: 保存音频
                        if context.audio_dump_file is not None:
                            logger.info('dump audio to file')
                            context.audio_dump_file.write(pcm_audio.tobytes())

                        # Clean up transcript
                        transcript = re.sub(r"<\|.*?\|>", "", transcript).strip()
                        if not context.config.stream:
                            if len(transcript) == 0:
                                # 如果 ASR 识别结果为空，则需要重新开启vad (类似SenseVoice)
                                logger.warning("Empty transcript, re-enabling VAD")
                                if context.shared_states:
                                    context.shared_states.enable_vad = True
                                return

                            # 输出转录结果 (先输出内容，标记未结束)
                            logger.info(f"Emitting transcript: {transcript}")
                            output = DataBundle(output_definition)
                            output.set_main_data(transcript)
                            output.add_meta('human_text_end', False)
                            output.add_meta('speech_id', context.speech_id)
                            yield output

                            # 然后输出结束标记
                            end_output = DataBundle(output_definition)
                            end_output.set_main_data('')
                            end_output.add_meta("human_text_end", True)
                            end_output.add_meta("speech_id", context.speech_id)
                            yield end_output

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    # 出现异常时也要重新开启VAD
                    if context.shared_states:
                        context.shared_states.enable_vad = True

                finally:
                    context.audio_buffer.clear()
            else:
                logger.warning("No audio data to process")

    def _call_qwen3_asr_api(self, context: Qwen3ASRContext, audio_file_path: str) -> str:
        """Call Qwen3-ASR API based on official example using dashscope SDK"""
        try:
            if dashscope is None:
                logger.error("dashscope not installed. Cannot call Qwen3-ASR API.")
                return ""

            logger.info(f"Calling Qwen3-ASR API with audio file: {audio_file_path}")

            # Initialize dashscope with API key and base URL
            dashscope.api_key = context.config.api_key
            dashscope.base_http_api_url = context.config.base_url

            logger.info(f"DashScope initialized with base_url: {context.config.base_url}")
            logger.info(f"API key (first 10 chars): {context.config.api_key[:10]}...")

            # Convert file path to file:// format
            absolute_path = os.path.abspath(audio_file_path)

            # Verify audio file exists and is readable
            if not os.path.exists(absolute_path):
                logger.error(f"Audio file does not exist: {absolute_path}")
                return ""

            file_size = os.path.getsize(absolute_path)
            logger.info(f"Audio file size: {file_size} bytes")

            file_url = f"file://{absolute_path}"

            # Prepare messages in the format from the official example
            messages = [
                {"role": "system", "content": [{"text": ""}]},  # 配置定制化识别的 Context
                {"role": "user", "content": [{"audio": file_url}]}
            ]

            logger.info(f"Audio file URL: {file_url}")
            logger.info(f"Sending request to qwen3-asr-flash model...")

            # Call the API using dashscope SDK (synchronous call)
            response = dashscope.MultiModalConversation.call(
                api_key=context.config.api_key,
                model=context.config.model_name,
                messages=messages,
                result_format="message",
                asr_options={
                    "language": context.config.language,
                    "enable_itn": context.config.enable_itn
                }
            )

            logger.info(f"API response received: {response}")

            # Check if API call was successful
            if response is None:
                logger.error("API call returned None response")
                return ""

            if "status_code" in response and response["status_code"] != 200:
                logger.error(f"API call failed with status code: {response.get('status_code')}")
                if "message" in response:
                    logger.error(f"Error message: {response['message']}")
                return ""

            # Extract transcription from response
            if response and "output" in response:
                output = response["output"]
                if "choices" in output and output["choices"]:
                    # Format similar to OpenAI API response
                    choice = output["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content_list = choice["message"]["content"]
                        for content in content_list:
                            if "text" in content:
                                transcript = content["text"]
                                logger.info(f"Extracted transcript: {transcript}")
                                return transcript
                elif "text" in output:
                    # Direct text format
                    transcript = output["text"]
                    logger.info(f"Direct transcript: {transcript}")
                    return transcript

            logger.warning("Could not extract transcript from response")
            return ""

        except Exception as e:
            import traceback
            logger.error(f"Error calling Qwen3-ASR API: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def _stream_qwen3_asr_api(self, context: Qwen3ASRContext, audio_file_path: str,
                              output_definition: DataBundleDefinition):
        """Call Qwen3-ASR API in streaming mode and emit partial results."""
        if dashscope is None:
            logger.error("dashscope not installed. Cannot call Qwen3-ASR API.")
            return

        logger.info(f"Streaming Qwen3-ASR API with audio file: {audio_file_path}")

        dashscope.api_key = context.config.api_key
        dashscope.base_http_api_url = context.config.base_url

        absolute_path = os.path.abspath(audio_file_path)
        if not os.path.exists(absolute_path):
            logger.error(f"Audio file does not exist: {absolute_path}")
            return

        messages = [
            {"role": "system", "content": [{"text": ""}]},
            {"role": "user", "content": [{"audio": f"file://{absolute_path}"}]}
        ]

        response_stream = dashscope.MultiModalConversation.call(
            api_key=context.config.api_key,
            model=context.config.model_name,
            messages=messages,
            result_format="message",
            asr_options={
                "language": context.config.language,
                "enable_itn": context.config.enable_itn
            },
            stream=True
        )

        last_text = ""
        finished = False
        try:
            for resp in response_stream:
                try:
                    output = resp.get("output", {})
                    choices = output.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    message = choice.get("message", {})
                    content_list = message.get("content", [])
                    finish_reason = choice.get("finish_reason")
                    transcript = ""
                    for content in content_list:
                        if "text" in content:
                            transcript = content["text"]
                    if transcript:
                        # 计算增量，避免重复发送
                        if transcript.startswith(last_text):
                            delta = transcript[len(last_text):]
                        else:
                            delta = transcript
                        last_text = transcript
                        delta = re.sub(r"<\|.*?\|>", "", delta).strip()
                        if len(delta) > 0:
                            output_bundle = DataBundle(output_definition)
                            output_bundle.set_main_data(delta)
                            output_bundle.add_meta('human_text_end', False)
                            output_bundle.add_meta('speech_id', context.speech_id)
                            yield output_bundle
                    if finish_reason == "stop":
                        finished = True
                        break
                except Exception as inner_e:
                    logger.error(f"Error parsing streaming response: {inner_e}")
                    continue
        except Exception as e:
            logger.error(f"Streaming call error: {e}")
        finally:
            # 结束标记
            end_output = DataBundle(output_definition)
            end_output.set_main_data('')
            end_output.add_meta("human_text_end", True)
            end_output.add_meta("speech_id", context.speech_id)
            yield end_output


    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        """Load handler configuration"""
        if isinstance(handler_config, Qwen3ASRConfig):
            config = handler_config
        else:
            config = Qwen3ASRConfig()
        self.config = config
        logger.info(f"Qwen3-ASR handler loaded with model: {config.model_name}")

    def create_context(self, session_context: SessionContext, handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        """Create a new handler context"""
        if isinstance(handler_config, Qwen3ASRConfig):
            config = handler_config
        else:
            config = self.config if hasattr(self, 'config') else Qwen3ASRConfig()

        context = Qwen3ASRContext(session_context.session_info.session_id)
        context.config = config
        context.shared_states = session_context.shared_states
        context.speech_id = session_context.session_info.session_id

        logger.info(f"Qwen3-ASR context created for session {session_context.session_info.session_id}")
        return context

    def destroy_context(self, context: HandlerContext):
        """Destroy handler context"""
        qwen3_context = cast(Qwen3ASRContext, context)

        # 关闭音频dump文件
        if qwen3_context.audio_dump_file is not None:
            qwen3_context.audio_dump_file.close()
            qwen3_context.audio_dump_file = None

        logger.info(f"Qwen3-ASR context destroyed for session {context.session_id}")

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        """Get handler details"""
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def shutdown(self):
        """Shutdown the handler"""
        logger.info("Shutting down Qwen3-ASR handler")
