import os
import re
from typing import Dict, Optional, cast

from abc import ABC
from loguru import logger
from openai import OpenAI, APIStatusError
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class TranslatorConfig(HandlerBaseConfigModel, BaseModel):
    # 使用 Qwen-MT 模型做专用翻译
    model_name: str = Field(default="qwen-mt-plus")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: Optional[str] = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    # 语义级纠偏所用的模型（通常是通用对话模型，如 qwen-plus）
    correction_model_name: Optional[str] = Field(default=None)
    enable_semantic_correction: bool = Field(default=False)
    # 与 Qwen-MT demo 对齐的语言名称
    source_lang: str = Field(default="Chinese")
    target_lang: str = Field(default="English")
    # 可选的文本纠偏规则：在翻译前对 ASR 文本做简单字符串替换
    # 例如：{"五声音图": "五十音图"}
    correction_rules: Dict[str, str] = Field(default_factory=dict)


class TranslatorContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[TranslatorConfig] = None
        self.client: Optional[OpenAI] = None
        self.input_text: str = ""


class HandlerTranslator(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TranslatorConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, TranslatorConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config when using HandlerTranslator'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[BaseModel] = None) -> HandlerContext:
        if not isinstance(handler_config, TranslatorConfig):
            handler_config = TranslatorConfig()
        context = TranslatorContext(session_context.session_info.session_id)
        context.config = handler_config
        context.client = OpenAI(
            api_key=handler_config.api_key,
            base_url=handler_config.api_url,
        )
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def _semantic_correct(self, text: str, context: TranslatorContext) -> str:
        """
        使用通用 LLM 对整句中文做轻量语义纠偏，并补充合适的标点符号。
        典型用法：修正常见 ASR 误识别的专有名词，补全句读，但保持语义不变。
        """
        cfg = context.config
        if not getattr(cfg, "enable_semantic_correction", False):
            return text
        model_name = getattr(cfg, "correction_model_name", None)
        if not model_name:
            return text

        prompt_system = (
            "你是中文语音识别结果的纠错助手。"
            "输入是一句已经被语音识别成文字的中文，可能有少量错别字、词语错误或缺少标点。"
            "你的任务是：在不改变原本语义和信息的前提下，对句子做轻微的纠正，并补充合适的标点，使其成为自然、通顺、语义合理的中文。"
            "不要翻译成其他语言，不要添加解释或额外内容，不要改变句子意思。"
            "只输出纠正并加上标点后的中文句子本身。"
        )

        try:
            completion = context.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": text},
                ],
            )
            corrected = completion.choices[0].message.content or ""
            corrected = corrected.strip()
            if corrected and corrected != text:
                logger.info(f"translator semantic corrected text: '{text}' -> '{corrected}'")
                return corrected
        except Exception as e:
            logger.warning(f"semantic correction failed, fallback to original text: {e}")
        return text

    @staticmethod
    def _apply_corrections(text: str, context: TranslatorContext) -> str:
        """在翻译前对识别文本做简单纠偏（基于配置的字符串替换）。"""
        rules = (context.config.correction_rules
                 if isinstance(context.config.correction_rules, dict)
                 else {})
        if not rules:
            return text
        original = text
        for src, tgt in rules.items():
            if not src:
                continue
            text = text.replace(src, tgt)
        if text != original:
            logger.info(f"translator corrected text: '{original}' -> '{text}'")
        return text

    def _build_messages(self, context: TranslatorContext, text: str):
        # Qwen-MT 官方 demo：只需要 user 内容，翻译配置通过 extra_body 传入
        return [
            {
                "role": "user",
                "content": text,
            }
        ]

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(TranslatorContext, context)
        if inputs.type != ChatDataType.HUMAN_TEXT:
            return

        text = inputs.data.get_main_data()
        if text is None:
            return
        text = str(text)

        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        # 累积一次完整发言的文本
        context.input_text += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        # 一次发言结束，调用翻译模型（整句翻译，保证语义流畅）
        source_text = re.sub(r"<\|.*?\|>", "", context.input_text).strip()
        # 先做语义级纠偏；字符串级纠偏可选，这里不再使用
        source_text = self._semantic_correct(source_text, context)
        context.input_text = ""
        if len(source_text) == 0:
            return

        # 调用翻译模型，输出英文 AVATAR_TEXT
        messages = self._build_messages(context, source_text)
        model_name = context.config.model_name
        translation_options = {
            "source_lang": context.config.source_lang,
            "target_lang": context.config.target_lang,
        }
        logger.info(
            f"translator input model={model_name} "
            f"source_lang={translation_options['source_lang']} "
            f"target_lang={translation_options['target_lang']} "
            f"text={source_text}"
        )

        try:
            completion = context.client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={
                    "translation_options": translation_options
                },
            )

            translated = ""
            if completion and completion.choices:
                translated = completion.choices[0].message.content or ""
            translated = re.sub(r"<\|.*?\|>", "", translated).strip()
            if translated:
                output = DataBundle(output_definition)
                output.set_main_data(translated)
                output.add_meta("avatar_text_end", False)
                output.add_meta("speech_id", speech_id)
                yield output

        except Exception as e:
            logger.error(e)
            error_text = ""
            if isinstance(e, APIStatusError):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    error_text = str(response["message"])
            if not error_text:
                error_text = "翻译服务暂时不可用，请稍后重试。"

            output = DataBundle(output_definition)
            output.set_main_data(error_text)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            yield output

        # 结束标记（英文翻译结束）
        end_output = DataBundle(output_definition)
        end_output.set_main_data("")
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
