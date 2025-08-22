import logging
from collections.abc import Generator
from typing import Optional, Union
from dify_plugin.errors.model import (
    CredentialsValidateFailedError
)
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
    I18nObject
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    PromptMessage, 
    PromptMessageTool, 
    UserPromptMessage,
    TextPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessageContentType
)
from .common import _Common


logger = logging.getLogger(__name__)


class SkyrouterLargeLanguageModel(OAICompatLargeLanguageModel, _Common):
    def _convert_files_to_text(self, messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        Convert any file content in messages to text descriptions to avoid validation issues
        """
        converted_messages = []
        
        for message in messages:
            if isinstance(message, UserPromptMessage) and isinstance(message.content, list):
                # Process multimodal content
                text_parts = []
                for content in message.content:
                    if isinstance(content, TextPromptMessageContent):
                        text_parts.append(content.data)
                    elif isinstance(content, ImagePromptMessageContent):
                        # Convert image to text description
                        if hasattr(content, 'url') and content.url:
                            text_parts.append(f"[Image file uploaded]: {content.url}")
                        else:
                            text_parts.append("[Image file uploaded]")
                    elif hasattr(content, 'type') and content.type == PromptMessageContentType.DOCUMENT:
                        # Handle document files like PDF
                        if hasattr(content, 'url') and content.url:
                            text_parts.append(f"[Document file uploaded]: {content.url}")
                        else:
                            text_parts.append("[Document file uploaded]")
                    else:
                        # Handle any other content types
                        if hasattr(content, 'url'):
                            text_parts.append(f"[File uploaded]: {content.url}")
                        else:
                            text_parts.append(str(content))
                
                # Create new text-only message
                converted_message = UserPromptMessage(content=" ".join(text_parts))
                converted_messages.append(converted_message)
            else:
                # Keep non-multimodal messages as is
                converted_messages.append(message)
        
        return converted_messages

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        
        # Convert any file content to text descriptions
        prompt_messages = self._convert_files_to_text(prompt_messages)
        # reasoning
        reasoning_params = {}
        reasoning_budget = model_parameters.pop('reasoning_budget', None)
        enable_thinking = model_parameters.pop('enable_thinking', None)
        if enable_thinking == 'dynamic':
            reasoning_budget = -1
        if reasoning_budget is not None:
            reasoning_params['max_tokens'] = reasoning_budget
        reasoning_effort = model_parameters.pop('reasoning_effort', None)
        if reasoning_effort is not None:
            reasoning_params['effort'] = reasoning_effort
        exclude_reasoning_tokens = model_parameters.pop('exclude_reasoning_tokens', None)
        if exclude_reasoning_tokens is not None:
            reasoning_params['exclude'] = exclude_reasoning_tokens
        if reasoning_params:
            model_parameters['reasoning'] = reasoning_params
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)
        return super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """
        If the reasoning response is from delta.get("reasoning") or delta.get("reasoning_content"),
        we wrap it with HTML think tag.

        :param delta: delta dictionary from LLM streaming response
        :param is_reasoning: is reasoning
        :return: tuple of (processed_content, is_reasoning)
        """

        content = delta.get("content") or ""
        # NOTE(hzw): OpenRouter uses "reasoning" instead of "reasoning_content".
        reasoning_content = delta.get("reasoning") or delta.get("reasoning_content")

        if reasoning_content:
            if not is_reasoning:
                content = "<think>\n" + reasoning_content
                is_reasoning = True
            else:
                content = reasoning_content
        elif is_reasoning and content:
            content = "\n</think>" + content
            is_reasoning = False
        return content, is_reasoning

    def _generate_block_as_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> Generator:
        resp = super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, False, user)
        yield LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=0,
                message=resp.message,
                usage=self._calc_response_usage(
                    model=model,
                    credentials=credentials,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                ),
                finish_reason="stop",
            ),
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)

    def validate_credentials(self, model: str, credentials: dict) -> None:        
        # Get model schema directly from YAML instead of calling get_model_schema
        try:
            model_config = self._load_model_from_yaml(model)
            
            print(f"Model config loaded from YAML: {model_config}")

            credentials['mode'] = model_config['model_properties']['mode']
            
            # If we successfully loaded the config, proceed with authentication
            # return self._validate_credentials_common(model, credentials) 
        except FileNotFoundError as e:
            # If model config file is not found, throw "model not supported" error
            raise CredentialsValidateFailedError(
                f"Model '{model}' is not supported."
            )

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        try:
            # Try to get model config from YAML first
            model_config = self._load_model_from_yaml(model)
            if model_config:
                base_model_schema = model_config
                base_model_schema_features = self._get_model_features_from_yaml(model)
                base_model_schema_model_properties = self._get_model_properties_from_yaml(model)
                base_model_schema_parameters_rules = self._get_parameter_rules_from_yaml(model)

                entity = AIModelEntity(
                    model=model,
                    label=I18nObject(zh_Hans=model, en_US=model),
                    model_type=ModelType.LLM,
                    features=list(base_model_schema_features),
                    fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
                    model_properties=dict(base_model_schema_model_properties.items()),
                    parameter_rules=list(base_model_schema_parameters_rules),
                    pricing=self._get_pricing_from_yaml(model),
                )

                return entity
        except Exception as e:
            print(f"Warning: Could not load model config from YAML: {e}")
