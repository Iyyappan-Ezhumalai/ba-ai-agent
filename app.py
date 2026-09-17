import streamlit as st
import pandas as pd
import pypdf
import docx
from PIL import Image
import io

# ==========================================
# 1. HELPER FUNCTIONS FOR FILE EXTRACTION
# ==========================================
def extract_content_from_file(uploaded_file):
    """Extracts text or image objects based on file type."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    # Text and Markdown files
    if file_type in ["txt", "md"]:
        return f"--- ATTACHMENT: {uploaded_file.name} ---\n" + uploaded_file.read().decode("utf-8") + "\n"
        
    # PDF Documents
    elif file_type == "pdf":
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = f"--- ATTACHMENT: {uploaded_file.name} (PDF) ---\n"
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text + "\n"
        
    # Word Documents (.docx)
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        text = f"--- ATTACHMENT: {uploaded_file.name} (DOCX) ---\n"
        text += "\n".join([para.text for para in doc.paragraphs])
        return text + "\n"
        
    # Spreadsheets (CSV & Excel)
    elif file_type in ["csv", "xlsx"]:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return f"--- ATTACHMENT: {uploaded_file.name} (Spreadsheet Data) ---\n" + df.to_string() + "\n"
        
    # Images (PNG, JPG, JPEG)
    elif file_type in ["png", "jpg", "jpeg"]:
        return Image.open(uploaded_file)
        
    return None

# ==========================================
# 2. SYSTEM PROMPTS FOR BA AGENTS
# ==========================================
PROMPTS = {
    "Requirement Gathering Agent": """
# ROLE & IDENTITY
You are a Senior Business Analyst AI specializing in Agile requirements elicitation and engineering.

# OBJECTIVE
Gather, structure, and refine business, functional, and non-functional requirements into precise BRDs and Agile User Stories.

# OUTPUT FORMATTING
1. Executive Summary & Assumptions
2. Requirements Traceability Matrix (RTM) Table (ID, Type, Requirement, Priority [MoSCoW], Rationale)
3. Agile User Stories with Acceptance Criteria (Given/When/Then Gherkin format)
4. Open Questions & Missing Signals
""",

    "Gap Analysis Agent": """
# ROLE & IDENTITY
You are a Strategic Business Analyst and Enterprise Architect AI specializing in Business Process Re-engineering (BPR).

# OBJECTIVE
Perform a comprehensive Gap Analysis comparing current processes (As-Is) against target states (To-Be).

# OUTPUT FORMATTING
1. Executive Gap Summary
2. As-Is vs. To-Be Capability Comparison Table (Capability, As-Is, To-Be, Gap Description, Gap Type)
3. Prioritized Gap Assessment & Remediation (Gap ID, Business Impact, Closure Options, Recommended Remediation)
4. Quick Wins & Strategic Initiatives
""",

    "Requirements Validation Agent": """
# ROLE & IDENTITY
You are a Lead Quality & Business Analysis Auditor AI specializing in requirements validation and edge-case testing.

# OBJECTIVE
Audit submitted requirements against the INVEST framework, identify edge cases, and refine acceptance criteria.

# OUTPUT FORMATTING
1. Quality Scorecard & INVEST Score (X/10)
2. Uncovered Edge Cases & Risks Table (Category, Scenario, Missing Handling, Impact)
3. Ambiguity Red Flags & Fixes
4. Refined Production-Ready User Story (Gherkin format)
5. Recommended Test Scenarios for QA
"""
}

# ==========================================
# 3. STREAMLIT APP UI & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Business Analyst AI Agent", layout="wide")

st.title("💼 Business Analyst AI Agent Dashboard")
st.caption("Automate Requirement Gathering, Gap Analysis, and Validation across multiple document formats.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")

provider = st.sidebar.selectbox(
    "Select Model Provider",
    ["Google Gemini", "Groq (Llama 3.3)", "OpenAI"]
)

# Dynamic Model Selection based on Provider
if provider == "Google Gemini":
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    )
elif provider == "Groq (Llama 3.3)":
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
else:  # OpenAI
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4o"]
    )

api_key = st.sidebar.text_input("Enter your API Key", type="password")

agent_type = st.sidebar.radio(
    "Select Agent Task",
    ["Requirement Gathering Agent", "Gap Analysis Agent", "Requirements Validation Agent"]
)

# Main Form Area
st.subheader(f"🤖 Active Agent: {agent_type}")

# File Uploader Widget
uploaded_files = st.file_uploader(
    "Upload Supporting Files (PDF, Word, Excel, CSV, Text, or Process Images)",
    type=["pdf", "docx", "txt", "md", "csv", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

user_text = st.text_area(
    "Enter Context or Instructions:",
    placeholder="Paste meeting notes, project brief, As-Is vs To-Be requirements, or draft stories...",
    height=200
)

# ==========================================
# 4. EXECUTION & API LOGIC
# ==========================================
if st.button("🚀 Run Analysis", type="primary"):
    if not api_key:
        st.error("Please enter a valid API Key in the sidebar.")
    elif not user_text and not uploaded_files:
        st.error("Please enter a text prompt or upload at least one file.")
    else:
        with st.spinner("Analyzing inputs and generating structured BA artifacts..."):
            system_prompt = PROMPTS[agent_type]
            text_context = f"{system_prompt}\n\n=== USER INPUT & CONTEXT ===\n{user_text}\n\n"
            images_list = []

            # Extract uploaded file content
            if uploaded_files:
                for file in uploaded_files:
                    content = extract_content_from_file(file)
                    if isinstance(content, str):
                        text_context += content
                    elif isinstance(content, Image.Image):
                        images_list.append(content)

            try:
                # Execution: Google Gemini
                if provider == "Google Gemini":
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    
                    # Combine text context and image objects for multimodal input
                    contents_payload = [text_context] + images_list
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents_payload
                    )
                    output_text = response.text

                # Execution: Groq (Llama 3.3)
                elif provider == "Groq (Llama 3.3)":
                    from openai import OpenAI
                    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text_context}
                        ]
                    )
                    output_text = response.choices[0].message.content

                # Execution: OpenAI
                elif provider == "OpenAI":
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text_context}
                        ]
                    )
                    output_text = response.choices[0].message.content

                # Display Results
                st.success("Analysis Complete!")
                st.markdown(output_text)

            except Exception as e:
                st.error(f"Error executing request: {e}")
