from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from mistralai import Mistral
import os
from dotenv import load_dotenv
import yaml
from scipy.spatial.distance import cosine
from utils import prompt_mistral
load_dotenv()


with open('prompts.yml', 'r') as f:
    prompts = yaml.safe_load(f)
    prompts = prompts['strategy_agent']


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


def state_initializer_strategy_agent(mapped_resume_fields_to_job_fields:dict):
    state:AgentState = {
        'mapped_resume_fields_to_job_fields': mapped_resume_fields_to_job_fields,
        'past_strategies': {}
    }

    return state

async def orchestrator(state:AgentState):
    state['is_repeated_strategy'] = None
    return state

async def strategy_generator(state:AgentState):
    llm_prompt = prompts['strategy_generator']
    current_nl_strategy = await prompt_mistral(llm_prompt.format(mappings = state['mapped_resume_fields_to_job_fields']))

    state['current_nl_strategy'] = current_nl_strategy

    return state

async def uniqueness_check(state:AgentState):
    if not state['past_strategies']:
        state['is_repeated_strategy']=False
        return state
    
    llm_prompt = prompts['uniqueness_check']
    current_nl_strategy = state['current_nl_strategy']
    past_strategies = state['past_strategies'].keys()

    model = "mistral-embed"

    current_nl_strategy_embeddings = fetch_embedding_creator(model=model, inputs=current_nl_strategy).data[0].embedding
    past_strategies_embeddings = {i: fetch_embedding_creator(model=model, inputs=i).data[0].embedding for i in past_strategies}

    cosine_similarities = {i: cosine(current_nl_strategy_embeddings, past_strategies_embeddings[i]) for i in past_strategies_embeddings.keys()}
    sorted_cosine_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)

    top_strategy_matches = [i for i, _ in sorted_cosine_similarities[:3]]

    repeated_or_not = await prompt_mistral(llm_prompt.format(current_strategy=current_nl_strategy, existing_strategies=top_strategy_matches))

    if "yes" in repeated_or_not.lower():
        state['is_repeated_strategy'] = True
        return state
    elif "no" in repeated_or_not.lower():
        state['is_repeated_strategy'] = False
        return state


def build_strategy_agent():
    graph = StateGraph(AgentState)

    graph.add_node('orchestrator', orchestrator)
    graph.add_node('strategy_generator', strategy_generator)
    graph.add_node('uniqueness_check', uniqueness_check)
    
    graph.add_edge('orchestrator', 'strategy_generator')
    graph.add_edge('strategy_generator', 'uniqueness_check')

    graph.set_entry_point('orchestrator')

    agent = graph.compile()

    return agent
    