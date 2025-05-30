import langgraph
from google import genai
from google.genai import types
import os
import asyncio
from typing_extensions import TypedDict
from input_preprocessing.input_preprocessing import page_image_to_md


async def generate_from_gemini(system_instruction, human_prompt):
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def _call_generate():
        return client.models.generate_content(
            model='gemini-2.0-flash',
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=[
                types.Part.from_text(human_prompt)
            ]
        )
    
    response = await asyncio.to_thread(_call_generate)
    await asyncio.sleep(10)
    return response.text

class AgentState(TypedDict):
    pagewise_chunked_resume_md: dict
    sectionwise_chunked_md: dict

def state_initialiser(fp):
    state:AgentState = {
        'pagewise_chunked_resume_md': page_image_to_md(fp),
        'sectionwise_chunked_md': {}
    }

    return state