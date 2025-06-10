import asyncio
from typing_extensions import TypedDict, Any
from langgraph.graph import StateGraph
from mistralai import Mistral
import os
from dotenv import load_dotenv
import yaml
from scipy.spatial.distance import cosine
load_dotenv()

with open('prompts.yml', 'r') as f:
    prompts = yaml.safe_load(f)
    prompts = prompts['strategy_agent']


async def prompt_mistral(prompt_text):
    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    messages = [{"role": "user", "content": prompt_text}]
    model = "mistral-large-latest"

    def sync_call():
        return client.chat.complete(model=model, messages=messages, temperature=0)
    
    response = await asyncio.to_thread(sync_call)
    return response.choices[0].message.content


def fetch_embedding_creator():
    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    return client.embeddings.create


def calculate_cosine(e1, e2):
    return 1-cosine(e1, e2)


class AgentState(TypedDict):
    mapped_resume_fields_to_job_fields: dict
    current_nl_strategy: str
    past_strategies: dict
    is_repeated_strategy: bool


def state_initializer(current_nl_strategy:str, mapped_resume_fields_to_job_fields:dict):
    state:AgentState = {
        'current_nl_strategy': current_nl_strategy,
        'mapped_resume_fields_to_job_fields': mapped_resume_fields_to_job_fields
    }

async def orchestrator(state:AgentState):
    state['is_repeated_strategy'] = None
    return state

async def strategy_generator(state:AgentState):
    llm_prompt = prompts['strategy_generator']
    current_nl_strategy = prompt_mistral(llm_prompt.format(mappings = state['mapped_resume_fields_to_job_fields']))

    state['current_nl_strategy'] = current_nl_strategy

    return state

async def uniqueness_check(state:AgentState):
    llm_prompt = prompts['uniqueness_check']
    current_nl_strategy = state['current_nl_strategy']
    past_strategies = state['past_strategies'].keys()

    model = "mistral-embed"

    current_nl_strategy_embeddings = fetch_embedding_creator(model=model, inputs=current_nl_strategy).data[0].embedding
    past_strategies_embeddings = {i: fetch_embedding_creator(model=model, inputs=i).data[0].embedding for i in past_strategies}

    cosine_similarities = {i: cosine(current_nl_strategy_embeddings, past_strategies_embeddings[i]) for i in past_strategies_embeddings.keys()}
    sorted_cosine_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)

    top_strategy_matches = [i for i, _ in sorted_cosine_similarities[:3]]

    repeated_or_not = prompt_mistral(llm_prompt.format(current_strategy=current_nl_strategy, existing_strategies=top_strategy_matches))

    if "yes" in repeated_or_not.lower():
        state['is_repeated_strategy'] = True
        return state
    elif "no" in repeated_or_not.lower():
        state['is_repeated_strategy'] = False
        return state


