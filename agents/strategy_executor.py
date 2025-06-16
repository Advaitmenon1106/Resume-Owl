from utils import fetch_embedding_creator, prompt_gemini, prompt_mistral
from typing_extensions import TypedDict
import yaml
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    strategy: str
    tools: dict
    response: str
    mapped_resume_fields_to_job_fields: dict
    jobs_json: list[dict]
    resume_json: dict
    relevant_job_fields: list
    resume_indexed: dict
    relevant_job_portions: list


with open('prompts.yml', 'r') as f:
    prompts = yaml.safe_load(f.read())['strategy_executor']


def strategy_executor_state_initializer(strategy:str, mapped_resume_fields_to_job_fields:dict, jobs_json:list[dict], resume_json:dict, relevant_job_fields:list):
    state:AgentState = {
        'strategy': strategy,
        'mapped_resume_fields_to_job_fields': mapped_resume_fields_to_job_fields,
        'jobs_json': jobs_json,
        'resume_json':resume_json,
        'relevant_job_fields': relevant_job_fields,
        "response": "",
        "resume_indexed": "",
        "relevant_job_portions": ""
    }

    return state


async def extract_target_keys(state:AgentState):
    llm_prompt = prompts['extract_target_keys']
    cs_keys = await prompt_mistral(llm_prompt.format(mapped_resume_fields_to_job_fields=state['mapped_resume_fields_to_job_fields'], strategy=state['strategy']))
    print(cs_keys)

    key_of_resume_json, key_of_job_json, unique_job_id_field = cs_keys.split(',')
    key_of_resume_json, key_of_job_json, unique_job_id_field = key_of_resume_json.strip(), key_of_job_json.strip(), unique_job_id_field.strip()

    print(f"Strategy: {state['strategy']}")
    print(f"Resume Key: {key_of_resume_json}")
    print(f"Job Key: {key_of_job_json}")

    resume_indexed = state['resume_json'][key_of_resume_json]
    relevant_job_portions = {}

    for i in state['jobs_json']:
        relevant_job_portions[i[unique_job_id_field]] = i[key_of_job_json]

    state['resume_indexed'] = resume_indexed
    state['relevant_job_portions'] = relevant_job_portions

    return state


def build_strategy_executor_agent():
    graph = StateGraph(AgentState)

    graph.add_node('extract_target_keys', extract_target_keys)

    graph.set_entry_point('extract_target_keys')

    agent = graph.compile()
    return agent