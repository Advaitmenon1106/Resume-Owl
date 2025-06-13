from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import asyncio
from mistralai import Mistral

load_dotenv()


async def prompt_gemini(prompt_text):
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def _call_generate():
        try:
            return client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    types.Part.from_text(text=prompt_text)
                ]
            )
        except:
            return client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[
                    types.Part.from_text(text=prompt_text)
                ]
            )

    response = await asyncio.to_thread(_call_generate)
    await asyncio.sleep(10)
    return response.text


async def prompt_mistral(prompt_text):
    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    model = "mistral-large-latest"
    messages = [{"role": "user", "content": prompt_text}]

    def sync_call():
        return client.chat.complete(model=model, messages=messages, temperature=0)

    try:
        response = await asyncio.to_thread(sync_call)
        return response.choices[0].message.content
    except Exception as e:
        print("Error from Mistral:", e)
        raise

def fetch_embedding_creator(inputs, model='mistral-embed'):
    client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
    return client.embeddings.create(model=model, inputs=inputs)