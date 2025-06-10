import asyncio
from typing_extensions import TypedDict, Any
from langgraph.graph import StateGraph
from input_preprocessing.input_preprocessing import convert_csv_to_json, convert_resume_to_json
from mistralai import Mistral
import os
from dotenv import load_dotenv
import yaml

load_dotenv()


with open('prompts.yml') as f:
    prompts = yaml.safe_load(f)
    prompts = prompts['preprocessing_agent']


async def prompt_mistral(prompt_text):
    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    model = "mistral-large-latest"
    messages = [{"role": "user", "content": prompt_text}]
    
    def sync_call():
        return client.chat.complete(model=model, messages=messages, temperature=0)
    
    response = await asyncio.to_thread(sync_call)
    return response.choices[0].message.content


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
    single_job_req = state['jobs_json'][0]
    input_prompt = prompts['relevant_job_field_extractor'].format(single_job_req=single_job_req)

    relevant_job_fields_cs = await prompt_mistral(input_prompt)
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

    for j_key in job_keys:
        imp_fields_cs = await prompt_mistral(input_prompt.format(j_key=j_key, resume_keys=resume_keys))
        imp_fields = imp_fields_cs.strip().split(',')
        imp_fields = [i.strip() for i in imp_fields]

        mapped_resume_fields_to_job_fields[j_key] = imp_fields
    
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