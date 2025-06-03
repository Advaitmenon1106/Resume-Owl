import asyncio
from typing_extensions import TypedDict, Any
from langgraph.graph import StateGraph
from input_preprocessing.input_preprocessing import convert_csv_to_json, convert_resume_to_json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import yaml

load_dotenv()


with open('../prompts.yml') as f:
    prompts = yaml.safe_load(f)


async def prompt_gemini(system_instruction, user_prompt):
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def _call_generate():
        try:
            return client.models.generate_content(
                model='gemini-2.0-flash',
                config=types.GenerateContentConfig(system_instruction=system_instruction),
                contents=[
                    types.Part.from_text(user_prompt)
                ]
            )
        except:
            return client.models.generate_content(
                model='gemini-1.5-flash',
                config=types.GenerateContentConfig(system_instruction=prompts['convert_to_md_system']),
                contents=[
                    types.Part.from_bytes(user_prompt)
                ]
            )

    response = await asyncio.to_thread(_call_generate)
    await asyncio.sleep(10)
    return response.text

class AgentState(TypedDict):
    jobs_json: list[dict]
    resume_json: dict[str, Any]
    mapped_resume_fields_to_job_fields:dict

def state_initialiser(fp_data, fp_resume):
    state:AgentState = {
        'jobs_json':convert_csv_to_json(fp_data),
        'resume_json': convert_resume_to_json(fp_resume),
        'mapped_resume_fields_to_job_fields': {}
    }

    return state

async def resumeKeys_to_jobKeys_mapper(state:AgentState):
    job_keys = list(state['jobs_json'][0].keys())
    resume_keys = [list(i.keys())[0] for i in state['resume_json']]

    print(resume_keys)

def graph_builder():
    graph = StateGraph(AgentState)

    graph.add_node('resumeKeys_to_jobKeys_mapper', resumeKeys_to_jobKeys_mapper)

    graph.set_entry_point('resumeKeys_to_jobKeys_mapper')

    agent = graph.compile()
    return agent