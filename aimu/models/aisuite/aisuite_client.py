from typing import Any, Iterator, Optional
from ..base_client import Model, ModelClient

import aisuite
import logging
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)


class AisuiteModel(Model):
    GPT_4O_MINI = "openai:gpt-4o-mini"
    GPT_4O = "openai:gpt-4o"

    CLAUDE_3_5_SONNET = "anthropic:claude-3-5-sonnet"
    
    ## TODO: re-enable GPT-5 models once issue with "max_completion_tokens" is resolved
    # GPT_5_NANO = "openai:gpt-5-nano"
    # GPT_5_MINI = "openai:gpt-5-mini"
    # GPT_5 = "openai:gpt-5"


class AisuiteClient(ModelClient):
    MODELS = AisuiteModel

    TOOL_MODELS = [
        MODELS.GPT_4O_MINI,
        MODELS.GPT_4O,
    ]

    THINKING_MODELS = []

    DEFAULT_GENERATE_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
    }

    def __init__(
        self,
        model: AisuiteModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()

        load_dotenv()

        self.ai_client = aisuite.Client()

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict[str, None]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        else:
            generate_kwargs = {**self.default_generate_kwargs, **generate_kwargs}

        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [{"role": "user", "content": prompt}]

        response = self.ai_client.chat.completions.create(
            self.model.value,
            messages,
            temperature=generate_kwargs["temperature"],
            max_tokens=generate_kwargs["max_new_tokens"],
        )

        return response.choices[0].message.content

    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [{"role": "user", "content": prompt}]

        response = self.ai_client.chat.completions.create(
            self.model.value,
            messages,
            stream=True,
            temperature=generate_kwargs["temperature"],
            max_tokens=generate_kwargs["max_new_tokens"],
        )

        message = {"role": next(response).choices[0].delta.role}
        content = ""

        for response_part in response:
            if response_part.choices[0].finish_reason is not None:
                break

            content += response_part.choices[0].delta.content
            yield response_part.choices[0].delta.content

        message["content"] = content

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = self.ai_client.chat.completions.create(
            self.model.value,
            self.messages,
            tools=tools,
            temperature=generate_kwargs["temperature"],
            max_tokens=generate_kwargs["max_new_tokens"],
        )

        if response.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                )

            self._handle_tool_calls(tool_calls, tools)

            ## tool calls for aisuite require the arguments to be a json string
            # TODO: find a better way. this makes messages inconsistent between Aisuite and other model clients
            self.messages[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                self.messages[-2]["tool_calls"][0]["function"]["arguments"]
            )

            response = self.ai_client.chat.completions.create(
                self.model.value,
                self.messages,
                tools=tools,
                temperature=generate_kwargs["temperature"],
                max_tokens=generate_kwargs["max_new_tokens"],
            )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[str]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = self.ai_client.chat.completions.create(
            self.model.value,
            self.messages,
            stream=True,
            tools=tools,
            temperature=generate_kwargs["temperature"],
            max_tokens=generate_kwargs["max_new_tokens"],
        )

        response_part = next(response)

        ## process tool calls
        if response_part.choices[0].delta.tool_calls:
            function_name = response_part.choices[0].delta.tool_calls[0].function.name

            function_arguments = ""
            while response_part := next(response):
                if response_part.choices[0].finish_reason is not None:
                    break

                function_arguments += response_part.choices[0].delta.tool_calls[0].function.arguments

            tool_calls = []  ## TODO: handle multiple tool calls
            tool_calls.append(
                {
                    "name": function_name,
                    "arguments": json.loads(function_arguments),
                }
            )

            self._handle_tool_calls(tool_calls, tools)

            ## tool calls for aisuite require the arguments to be a json string
            # TODO: find a better way. this makes messages inconsistent between Aisuite and other model clients
            self.messages[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                self.messages[-2]["tool_calls"][0]["function"]["arguments"]
            )

            response = self.ai_client.chat.completions.create(
                self.model.value,
                self.messages,
                stream=True,
                tools=tools,
                temperature=generate_kwargs["temperature"],
                max_tokens=generate_kwargs["max_new_tokens"],
            )

            response_part = next(response)

        message = {"role": response_part.choices[0].delta.role}
        content = ""

        for response_part in response:
            if response_part.choices[0].finish_reason is not None:
                break

            content += response_part.choices[0].delta.content
            yield response_part.choices[0].delta.content

        message["content"] = content
        self.messages.append(message)
