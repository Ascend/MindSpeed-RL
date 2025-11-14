import json
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class OpenAIFunctionPropertySchema(BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(type="object", properties={}, required=[])
    )
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
