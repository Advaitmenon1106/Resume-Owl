import requests
import tempfile
import os
from dotenv import load_dotenv
import pdf2image
from PIL import Image
import yaml
import base64
from io import BytesIO
import asyncio
import json
import re
import pandas as pd
from mistralai import Mistral

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
    os.remove(os.environ['OUTPUT_PDF_PATH'])

    return page_images


async def send_image_to_mistral(images: list[Image.Image]):
    llm = Mistral(os.environ['MISTRAL_API_KEY'])
    model = "pixtral-12b-2409"

    content = [{"type": "text", "text": prompts["convert_to_md_system"]}]
    for img in images:
        b64_img = pil_to_base64(img, format="JPEG")
        content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64_img}"
        })

    messages = [{"role": "user", "content": content}]

    response = await asyncio.to_thread(
        lambda: llm.chat.complete(model=model, messages=messages)
    )

    return response.choices[0].message.content


async def convert_images_to_json(image_array: list[Image.Image], batch_size: int = 3):
    semaphore = asyncio.Semaphore(5)

    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    async def limited_task(image_batch):
        async with semaphore:
            return await send_image_to_mistral(image_batch)

    tasks = [limited_task(batch) for batch in chunks(image_array, batch_size)]
    return await asyncio.gather(*tasks)


async def page_image_to_json(fp):
    images_array = chunk_pdf_to_images(fp)
    md_pages = await convert_images_to_json(images_array)
    return md_pages


def clean_and_parse_json(llm_output):
    # Remove Markdown fences if they exist
    cleaned = re.sub(r"^```json\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def convert_resume_to_json(fp):
    print("Converting the PDF to JSON")
    res = asyncio.run(page_image_to_json(fp))
    
    # Handle multiple chunks
    parsed = []
    if isinstance(res, list):
        for item in res:
            parsed.append(clean_and_parse_json(item))
    else:
        parsed = clean_and_parse_json(res)

    final_json = parsed[0]
    final_json.extend(parsed[1:])
    print(final_json[0])
    final_json = {k: v for d in final_json for k, v in d.items()}

    with open('output.json', 'w') as f:
        json.dump(parsed, f, indent=2)
    
    return final_json


def convert_csv_to_json(fp):
    return pd.read_csv(fp, nrows=1000).to_dict(orient='records')

# with open('sample_inputs/jobs_json.json', 'w') as f:
#     json.dump(convert_csv_to_json('sample_inputs/job_descriptions.csv'), f, indent=2)