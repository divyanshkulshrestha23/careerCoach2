import gradio as gr
import trafilatura
from processor import resume_polish, cover_letter, career_coach, extract_text_from_pdf
from filestore import store_resume_chunks, get_stored_resume_text
from scoring import match_resume, format_report


def fetch_job_context(input_string):
    clean_input = input_string.strip()
    if clean_input.startswith("http://") or clean_input.startswith("https://"):
        downloaded = trafilatura.fetch_url(clean_input)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text
        return "Error: Could not extract text from the URL. Please paste the raw description instead."
    return clean_input


def unified_upload_and_polish(file, position, instructions):
    """Handles PDF extraction, database storage, and resume polishing in one click."""
    if file is None:
        return "Please upload a PDF resume."

    # 1. Extract the text using the YOLO/Tesseract pipeline
    parsed_sections = extract_text_from_pdf(file.name)

    # 2. Silently store it in ChromaDB so Tabs 2 and 3 can use it later
    store_resume_chunks(parsed_sections)

    # 3. Combine the text and send it directly to the Polisher LLM
    full_text = "\n\n".join([f"--- {k} ---\n{v}" for k, v in parsed_sections.items()])
    return resume_polish(position, instructions, full_text)


def generate_cover_letter_wrapper(comp, pos, desc, res):
    actual_desc = fetch_job_context(desc)
    return cover_letter(comp, pos, actual_desc, res)


def career_coach_wrapper(desc, res, pos):
    actual_desc = fetch_job_context(desc)
    return career_coach(actual_desc, res, pos)


def resume_match_wrapper(desc, res):
    actual_desc = fetch_job_context(desc)
    resume_text = res.strip() or get_stored_resume_text()
    if not resume_text:
        return "Upload a resume in Tab 1, or paste your resume text here, then try again."
    return format_report(match_resume(resume_text, actual_desc))


custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

force_dark = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Build a unified interface using Gradio Blocks
with gr.Blocks(title="AI Career Assistant") as app:
    gr.Markdown("# AI Career Assistant")
    gr.Markdown("Your all-in-one tool for polishing resumes, drafting cover letters, and getting career advice.")

    # --- TAB 1: The Unified Polisher ---
    with gr.Tab("1. Resume Polisher"):
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload Resume PDF", file_types=[".pdf"])
                pos_in = gr.Textbox(label="Target Position", placeholder="e.g., Software Engineer Intern...")
                inst_in = gr.Textbox(label="Special Instructions (Optional)",
                                     placeholder="e.g., Make it sound more technical, highlight leadership...", lines=2)
                polish_btn = gr.Button("Extract & Polish Resume", variant="primary")
            with gr.Column():
                polish_out = gr.Textbox(label="Polished Resume Output", lines=20)

        polish_btn.click(fn=unified_upload_and_polish, inputs=[pdf_input, pos_in, inst_in], outputs=[polish_out])

    # --- TAB 2: Cover Letter ---
    with gr.Tab("2. Cover Letter Generator"):
        with gr.Row():
            with gr.Column():
                comp_in = gr.Textbox(label="Company Name", placeholder="Enter the name of the company...")
                c_pos_in = gr.Textbox(label="Position Name", placeholder="Enter the name of the position...")
                c_desc_in = gr.Textbox(label="Job Description",
                                       placeholder="Paste the job description or a URL here...",
                                       lines=5)
                c_res_in = gr.Textbox(label="Manual Resume Input (Optional)",
                                      placeholder="Leave blank to use the PDF you uploaded in Tab 1...", lines=3)
                cover_btn = gr.Button("Generate Cover Letter")
            with gr.Column():
                cover_out = gr.Textbox(label="Customized Cover Letter", lines=15)

        cover_btn.click(fn=generate_cover_letter_wrapper, inputs=[comp_in, c_pos_in, c_desc_in, c_res_in],
                        outputs=[cover_out])

    # --- TAB 3: Career Advice ---
    with gr.Tab("3. Career Advisor"):
        with gr.Row():
            with gr.Column():
                a_pos_in = gr.Textbox(label="Position Applied For", placeholder="Enter the position...")
                a_desc_in = gr.Textbox(label="Job Description",
                                       placeholder="Paste the job description or a URL here...", lines=5)
                a_res_in = gr.Textbox(label="Manual Resume Input (Optional)",
                                      placeholder="Leave blank to use the PDF you uploaded in Tab 1...", lines=3)
                advice_btn = gr.Button("Get Career Advice")
            with gr.Column():
                advice_out = gr.Textbox(label="Advice", lines=15)

        advice_btn.click(fn=career_coach_wrapper, inputs=[a_desc_in, a_res_in, a_pos_in], outputs=[advice_out])

    # --- TAB 4: Skill match score (local, no LLM) ---
    with gr.Tab("4. Resume Match"):
        with gr.Row():
            with gr.Column():
                m_desc_in = gr.Textbox(label="Job Description",
                                       placeholder="Paste the job description or a URL here...", lines=5)
                m_res_in = gr.Textbox(label="Manual Resume Input (Optional)",
                                      placeholder="Leave blank to use the PDF you uploaded in Tab 1...", lines=3)
                match_btn = gr.Button("Check Skill Match")
            with gr.Column():
                match_out = gr.Textbox(label="Match Report", lines=12)

        match_btn.click(fn=resume_match_wrapper, inputs=[m_desc_in, m_res_in], outputs=[match_out])

if __name__ == "__main__":
    app.launch(
        theme=custom_theme, js=force_dark)
