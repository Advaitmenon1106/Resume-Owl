import requests
import tempfile
import os
from dotenv import load_dotenv
import pdf2image
from PIL import Image
from google import genai
from google.genai import types
import yaml
import base64
from io import BytesIO
import asyncio
import json
import re

load_dotenv()

with open('prompts.yml', 'r') as f:
    prompts = yaml.safe_load(f)
    prompts = prompts['input_preprocessing']

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def document_to_pdf(input_fp):
    GOTENBERG_URL = os.environ['GOTENBERG_URL']

    with open(input_fp, 'rb') as f:
        files = {
            'files': ('sample.docx', f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        }

        data = {
            'landscape': 'false',
            'paperSize': 'A4',
        }

        response = requests.post(GOTENBERG_URL, files=files, data=data)

        if response.ok:
            # Create a temp file for the output PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(response.content)
                temp_path = temp_pdf.name
            print(f"PDF created at temporary path: {temp_path}")
            os.environ['OUTPUT_PDF_PATH'] = temp_path
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


def process_input(input_fp: str) -> str | None:
    """
    If input is .docx, convert to PDF and copy it to a temp location.
    If input is already .pdf, copy to a temp location.
    Returns path to a local PDF file or None on error.
    """
    if input_fp.endswith(".docx"):
        return document_to_pdf(input_fp)

    elif input_fp.endswith(".pdf"):
        with open(input_fp, "rb") as f:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(f.read())
                os.environ['OUTPUT_PDF_PATH'] = temp_pdf

    else:
        raise Exception("[ERROR] Unsupported file type")


def chunk_pdf_to_images(input_fp):
    # Save the PDF to a tmp directory

    process_input(input_fp) # os.environ['OUTPUT_PDF_PATH'] is defined here

    page_images = pdf2image.convert_from_path(os.environ['OUTPUT_PDF_PATH'])

    return page_images


async def send_image_to_gemini(img: Image.Image):
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    b64_image = pil_to_base64(img)

    def _call_generate():
        return client.models.generate_content(
            model='gemini-2.0-flash',
            config=types.GenerateContentConfig(system_instruction=prompts['convert_to_md_system']),
            contents=[
                types.Part.from_bytes(data=b64_image, mime_type='image/jpeg')
            ]
        )

    response = await asyncio.to_thread(_call_generate)
    await asyncio.sleep(10)
    return response.text


async def convert_images_to_markdown(image_array: list[Image.Image]):
    semaphore = asyncio.Semaphore(5)

    async def limited_task(img):
        async with semaphore:
            return await send_image_to_gemini(img)

    tasks = [limited_task(img) for img in image_array]

    return await asyncio.gather(*tasks)


async def page_image_to_md(fp):
    images_array = chunk_pdf_to_images(fp)
    md_pages = await convert_images_to_markdown(images_array)
    return md_pages


def clean_and_parse_json(llm_output):
    # Remove Markdown fences if they exist
    cleaned = re.sub(r"^```json\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


if __name__ == "__main__":
    res = asyncio.run(page_image_to_md('sample_inputs/Advait-Menon_May_2025_Resume.docx'))
    
    # Handle multiple chunks
    parsed = []
    if isinstance(res, list):
        for item in res:
            parsed.append(clean_and_parse_json(item))
    else:
        parsed = clean_and_parse_json(res)

    print(json.dumps(parsed, indent=2))
    
    with open('output.json', 'w') as f:
        json.dump(parsed, f, indent=2)
