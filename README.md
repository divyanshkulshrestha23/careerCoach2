# AI Career Assistant

An end-to-end, RAG-backed web application designed to automate and optimize the job application process. Built with Python, this tool leverages local vector databases, advanced layout detection, and Google's Gemini 3.6 Flash to parse technical resumes, generate targeted cover letters, and provide actionable career advice.

## System Architecture & Tech Stack

This application utilizes a hybrid local/cloud architecture to minimize latency and protect user data:
* **Frontend:** Gradio (Responsive Web UI)
* **Computer Vision / OCR:** YOLO (Document Layout Detection) & Tesseract OCR
* **Vector Database (RAG):** ChromaDB (Local persistent storage)
* **LLM Engine:** Google Gemini 2.0 Flash (`google-genai` SDK)
* **Web Scraping:** Trafilatura (Automated job description extraction)

## Key Features

* **Intelligent Document Parsing:** Utilizes a custom YOLO model to segment PDF resumes into logical blocks (Education, Experience, Skills) before extracting text via Tesseract OCR.
* **Retrieval-Augmented Generation (RAG):** Parses and embeds resume chunks into a local ChromaDB instance, allowing the system to query only the most relevant bullet points for specific job descriptions.
* **Automated URL Scraping:** Users can paste a job posting URL directly into the app; Trafilatura instantly scrapes and cleans the target text.
* **One-Click Polishing:** Employs the X-Y-Z action formula to rewrite raw resume bullet points into highly quantifiable, Big-Tech-optimized statements.

## Quick Start Installation

1. **Clone the repository:**
   `git clone https://github.com/yourusername/careerCoach2.git`
2. **Install system dependencies:**
   * Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and add it to your system PATH.
   * Install [Poppler](https://poppler.freedesktop.org/) (required for `pdf2image`).
3. **Install Python dependencies:**
   `pip install -r requirements.txt`
4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google API key:
   `GEMINI_API_KEY=your_api_key_here`
5. **Run the application:**
   `python app.py`

## Usage

Navigate to the local Gradio URL (typically `http://127.0.0.1:7860`). Upload your PDF resume in the **Resume Polisher** tab to automatically populate the local vector database, then freely generate custom cover letters and gap analyses in the subsequent tabs.
