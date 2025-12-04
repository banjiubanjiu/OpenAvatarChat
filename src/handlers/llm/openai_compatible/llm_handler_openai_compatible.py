

import os
import re
from typing import Dict, Optional, cast
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from openai import APIStatusError, OpenAI
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from handlers.llm.openai_compatible.chat_history_manager import ChatHistory, HistoryMessage


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    history_length: int = Field(default=20)
    # Router configuration: use a lightweight model (e.g. tongyi-intent-detect-v3)
    # to decide whether this turn needs vision (image) or plain text.
    router_enabled: bool = Field(default=False)
    router_model_name: Optional[str] = Field(default=None)
    # Optional override models for different routes; fall back to model_name when not set.
    text_model_name: Optional[str] = Field(default=None)
    vision_model_name: Optional[str] = Field(default=None)


class LLMContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.model_name = None
        self.system_prompt = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.history = None
        self.enable_video_input = False


class HandlerLLM(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.config = handler_config
        context.model_name = handler_config.model_name
        context.system_prompt = {'role': 'system', 'content': handler_config.system_prompt}
        context.api_key = handler_config.api_key
        context.api_url = handler_config.api_url
        context.enable_video_input = handler_config.enable_video_input
        context.history = ChatHistory(history_length=handler_config.history_length)
        context.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=context.api_key,
            base_url=context.api_url,
        )
        return context
    
    def start_context(self, session_context, handler_context):
        pass

    def _route_model(self, context: LLMContext, chat_text: str) -> tuple[str, bool]:
        """
        Decide which model to use for this turn and whether to include image input.

        Returns:
            (target_model_name, use_image)
        """
        cfg: Optional[LLMConfig] = context.config
        # Backwards-compatible fallback: single model, include image if available.
        if cfg is None:
            use_image = context.enable_video_input and context.current_image is not None
            return context.model_name, use_image

        text_model = cfg.text_model_name or cfg.model_name
        vision_model = cfg.vision_model_name or cfg.model_name

        # If router not enabled, simply use vision model when image is available.
        if not cfg.router_enabled:
            use_image = context.enable_video_input and context.current_image is not None
            target_model = vision_model if use_image else text_model
            return target_model, use_image

        router_model = cfg.router_model_name or text_model

        # Use a lightweight classifier model (e.g. tongyi-intent-detect-v3) to decide
        # whether this turn requires vision. We treat it as a simple classifier:
        # it should answer only "VISION" or "TEXT".
        tag = "TEXT"
        try:
            completion = context.client.chat.completions.create(
                model=router_model,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个意图分类器。"
                            "如果用户这句话需要你根据摄像头画面、图片或视频来回答问题，"
                            "例如：让你“看一下画面/照片/视频里有什么”，"
                            "询问“我现在长什么样/好不好看/穿着怎样/你猜我多大”，"
                            "或者问“我手上是什么/桌上有什么/画面里这些东西是什么”等，"
                            "都判为 VISION。普通聊天、情绪表达或与画面无关的问题判为 TEXT。"
                            "只回答 VISION 或 TEXT，不要输出其它内容。"
                        ),
                    },
                    {"role": "user", "content": chat_text},
                ],
            )
            tag = (completion.choices[0].message.content or "").strip().upper()
        except Exception as e:
            logger.warning(f"Vision router failed, fallback to TEXT: {e}")

        use_image = (
            tag == "VISION"
            and context.enable_video_input
            and context.current_image is not None
        )
        target_model = vision_model if use_image else text_model
        logger.info(
            f"llm router model={router_model} tag={tag} "
            f"use_image={use_image} target_model={target_model}"
        )
        return target_model, use_image

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)
        text = None
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            context.current_image = inputs.data.get_main_data()
            return
        elif inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            return
        target_model, use_image = self._route_model(context, chat_text)
        logger.info(f'llm input {target_model} use_image={use_image} {chat_text} ')
        current_content = context.history.generate_next_messages(
            chat_text,
            [context.current_image] if use_image else [],
        )
        logger.debug(f'llm input {context.model_name} {current_content} ')
        try:
            completion = context.client.chat.completions.create(
                model=target_model,  # 按路由选择文本或多模态模型
                messages=[
                    context.system_prompt,
                ] + current_content,
                stream=True,
                stream_options={"include_usage": True}
            )
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            for chunk in completion:
                if (chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content):
                    output_text = chunk.choices[0].delta.content
                    context.output_texts += output_text
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    yield output
            context.history.add_message(HistoryMessage(role="human", content=chat_text))
            context.history.add_message(HistoryMessage(role="avatar", content=context.output_texts))
        except Exception as e:
            logger.error(e)
            if (isinstance(e, APIStatusError)):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    response = f"{response['message']}"
            output_text = response 
            output = DataBundle(output_definition)
            output.set_main_data(output_text)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            yield output
        context.input_texts = ''
        context.output_texts = ''
        logger.info('avatar text end')
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
