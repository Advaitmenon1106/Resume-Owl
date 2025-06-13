from utils import fetch_embedding_creator, prompt_gemini, prompt_mistral
from typing_extensions import TypedDict

class AgentState(TypedDict):
    strategy: str
    tools: dict
    response: str
    mapped_resume_fields_to_job_fields: dict
    jobs_json: list[dict]
    resume_json: dict

def strategy_executor_state_initializer(strategy:str, mapped_resume_fields_to_job_fields:dict, jobs_json:list[dict], resume_json:dict):
    tools = {
        "embedder": {
            "function_name": "fetch_embedding_creator",
            "async": False,
            "description": "Embeds a single text or a set/list/array of texts",
            "parameters": {"inputs": "str or list[str]", "model": "mistral-embed, if nothing else is specified"},
            "returns": "EmbeddingResponse, using client.embeddings.create(model=model, inputs=inputs)"
        },
        "invoke_mistral": {
            "function_name": "prompt_mistral",
            "async": True,
            "description": "Uses Mistral's chat completion/generation module to return the LLM response as a string",
            "parameters": {"prompt_text": "str prompt"},
            "returns": "CoroutineType[Any, Any, OptionalNullable[AssistantMessageContent]]"
        },
        "invoke_gemini": {
            "function_name": "prompt_gemini",
            "async": True,
            "description": "Uses Gemini's chat completion/generation module to return the LLM response as a string",
            "parameters": {"prompt_text": "str prompt"},
            "returns": "CoroutineType[Any, Any, str | None]"
        }
    }

    state:AgentState = {
        'strategy': strategy,
        'tools': tools,
        'mapped_resume_fields_to_job_fields': mapped_resume_fields_to_job_fields,
        'jobs_json': jobs_json,
        'resume_json':resume_json,
        "response": ""
    }

    return state

async def execute_strategy(state:AgentState):
    pass