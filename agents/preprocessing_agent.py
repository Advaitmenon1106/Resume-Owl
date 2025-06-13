import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from input_preprocessing.input_preprocessing import convert_csv_to_json, convert_resume_to_json
from dotenv import load_dotenv
import yaml
import json
from utils import prompt_gemini, prompt_mistral


load_dotenv()


with open('prompts.yml') as f:
    prompts = yaml.safe_load(f)
    prompts = prompts['preprocessing_agent']


class AgentState(TypedDict):
    jobs_json: list[dict]
    resume_json: dict
    mapped_resume_fields_to_job_fields:dict
    relevant_job_fields:list


def state_initialiser(fp_data, fp_resume):
    state:AgentState = {
        'jobs_json':convert_csv_to_json(fp_data),
        'resume_json': convert_resume_to_json(fp_resume),
    }

    return state


async def relevant_job_field_extractor(state:AgentState):
    await asyncio.sleep(1.0)
    single_job_req = state['jobs_json'][0]
    input_prompt = prompts['relevant_job_field_extractor'].format(single_job_req=single_job_req)

    relevant_job_fields_cs = await prompt_gemini(input_prompt)
    relevant_job_fields = relevant_job_fields_cs.strip().split(',')
    relevant_job_fields = [i.strip() for i in relevant_job_fields]
    state['relevant_job_fields'] = relevant_job_fields

    return state


async def resumeKeys_to_jobKeys_mapper(state:AgentState):
    job_keys = state['relevant_job_fields']
    resume_keys = []
    input_prompt = prompts['resumeKeys_to_jobKeys_mapper']

    for i in state['resume_json']:
        resume_keys.append(list(i.keys())[0])

    mapped_resume_fields_to_job_fields = {}

    json_mappings = await prompt_mistral(prompt_text=input_prompt.format(job_keys=job_keys, resume_keys=resume_keys))
    json_mappings = json_mappings.strip("'").strip('"').strip('```').strip('json')
    mapped_resume_fields_to_job_fields = json.loads(json_mappings)
    
    state['mapped_resume_fields_to_job_fields'] = mapped_resume_fields_to_job_fields
    return state


def graph_builder():
    graph = StateGraph(AgentState)

    graph.add_node('relevant_job_field_extractor', relevant_job_field_extractor)
    graph.add_node('resumeKeys_to_jobKeys_mapper', resumeKeys_to_jobKeys_mapper)

    graph.add_edge('relevant_job_field_extractor', 'resumeKeys_to_jobKeys_mapper')

    graph.set_entry_point('relevant_job_field_extractor')

    agent = graph.compile()
    return agent