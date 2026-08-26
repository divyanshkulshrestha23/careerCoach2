import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from filestore import query_relevant_experience  # Import the RAG query function
import cv2
import pytesseract
from pdf2image import convert_from_path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyansh\AppData\Local\Tesseract-OCR\tesseract.exe'


def extract_text_from_pdf(pdf_path):
    """Parses PDF resumes using YOLO layout detection and Tesseract OCR."""
    images = convert_from_path(pdf_path, poppler_path=r"C:\Users\Divyansh\AppData\Local\poppler\Library\bin")
    model_path = hf_hub_download(repo_id="Armaggheddon/yolo26-document-layout", filename="yolo26n_doc_layout.pt")
    model = YOLO(model_path)

    parsed_sections = {}

    for page_idx, img in enumerate(images):
        img_path = f"temp_page_{page_idx}.png"
        img.save(img_path)
        cv_img = cv2.imread(img_path)

        results = model(cv_img)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            crop = cv_img[y1:y2, x1:x2]
            text = pytesseract.image_to_string(crop).strip()

            if text:
                parsed_sections[label] = parsed_sections.get(label, "") + "\n" + text

        if os.path.exists(img_path):
            os.remove(img_path)

    return parsed_sections


#######------------- LLM -------------####

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GEMINI_API_KEY)

######-------------Resume Polish-------------######

temp = """
Given the resume excerpt: '{resume_content}', polish it based on the following instructions: 

{polish_prompt} for the {position_name} position. Ensure it highlights quantifiable impact.
"""

pt = PromptTemplate(
    input_variables=["resume_content", "polish_prompt", "position_name"],
    template=temp)

prompt_to_Gemini = pt | llm


def resume_polish(position, polish_prompt, content):
    result = prompt_to_Gemini.invoke({
        "resume_content": content,
        "polish_prompt": polish_prompt,
        "position_name": position
    })
    if isinstance(result.content, list):
        return result.content[0].get("text", "")
    return result.content


######-------------Cover Letter-------------######

# UPDATED PROMPT: Now explicitly mentions it is receiving "targeted excerpts"
temp1 = ("Generate a customized cover letter using the company name: {company_name}, the position applied for: {"
         "position_name}, and the job description: {job_description}. Ensure the cover letter highlights my "
         "qualifications as detailed in the following relevant resume excerpts: {resume_content}. Adapt the content "
         "carefully to avoid including experiences not present in my resume but mentioned in the job description. "
         "Emphasize the alignment between my existing skills and the requirements of the role.")

pt1 = PromptTemplate(
    input_variables=["company_name", "position_name", "job_description", "resume_content"],
    template=temp1)

prompt_to_Gemini1 = pt1 | llm


def cover_letter(company, position, description, content=""):
    # 1. Query ChromaDB to find resume chunks matching the job description
    targeted_context = query_relevant_experience(description)

    # 2. Fallback to manually pasted content if the database is empty
    final_context = targeted_context if targeted_context and "No resume data found" not in targeted_context else content

    result = prompt_to_Gemini1.invoke({
        "company_name": company,
        "position_name": position,
        "job_description": description,
        "resume_content": final_context,
    })
    if isinstance(result.content, list):
        return result.content[0].get("text", "")
    return result.content


######-------------Career Coach-------------######

# UPDATED PROMPT: Tailored to analyze targeted chunks for precise gap analysis
temp2 = ("Considering the job description: {job_description}, and the targeted resume excerpts provided: "
         "{resume_content}, identify areas for enhancement. Offer specific suggestions on how to improve these aspects "
         "to better match the job requirements and increase the likelihood of being selected for the position of {"
         "position_applied}.")

pt2 = PromptTemplate(
    input_variables=["job_description", "resume_content", "position_applied"],
    template=temp2)

prompt_to_Gemini2 = pt2 | llm


def career_coach(description, content="", position=""):
    # 1. Query ChromaDB to find resume chunks matching the job description
    targeted_context = query_relevant_experience(description)

    # 2. Fallback to manually pasted content if the database is empty
    final_context = targeted_context if targeted_context and "No resume data found" not in targeted_context else content

    result = prompt_to_Gemini2.invoke({
        "job_description": description,
        "resume_content": final_context,
        "position_applied": position,
    })
    if isinstance(result.content, list):
        return result.content[0].get("text", "")
    return result.content
