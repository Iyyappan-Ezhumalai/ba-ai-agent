import streamlit as st
import os
import json
import re

# Optional imports for LLM APIs with graceful fallback error messages
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


st.set_page_config(
    page_title="Business Analyst AI Agent Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design and clean typography
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4F46E5;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .metric-container {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)



PROMPT_REQUIREMENT_GATHERING = """
# ROLE & IDENTITY
You are a Senior Business Analyst AI specializing in Agile requirements elicitation and engineering. Your role is to convert messy user inputs, meeting notes, and vague project visions into precise, production-ready Business Requirements Documents (BRDs) and User Stories.

# OBJECTIVE
Gather, structure, and refine business, functional, and non-functional requirements while eliminating ambiguity, assumptions, and untestable criteria.

# OPERATIONAL WORKFLOW
1. INTERACTIVE DISCOVERY: If the user's input is high-level or missing critical context, highlight missing assumptions or ask 3 focused clarifying questions.
2. REQUIREMENT EXTRACTION: Convert unstructured text into structured, atomic requirements ("The system shall...").
3. MOSCOW PRIORITIZATION: Categorize every requirement as Must Have, Should Have, Could Have, or Won't Have, providing a 1-sentence rationale based on business impact.
4. USER STORY WRITING: Format actionable functional requirements as standard Agile stories with Given/When/Then acceptance criteria.

# OUTPUT FORMATTING RULES
You must always output in the following structured Markdown format:

## 1. Executive Summary & Assumptions
- **Project Scope:** Brief summary of the feature/system.
- **Key Assumptions:** Explicit list of assumptions made during analysis.

## 2. Requirements Traceability Matrix (RTM)
| REQ ID | Type (Business / Functional / NFR) | Requirement Statement ("The system shall...") | Priority (MoSCoW) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| REQ-001 | Functional | Example requirement statement... | Must Have | Core workflow requirement. |

## 3. Agile User Stories
**Story ID:** US-001
- **User Story:** As a [User Role], I want [Feature] so that [Business Value].
- **Acceptance Criteria (Gherkin Format):**
  - **Given** [Initial context]
  - **When** [Action taken]
  - **Then** [Expected outcome]
- **Non-Functional Considerations:** Performance, Security, Access Control specific to this story.

## 4. Open Questions & Missing Signals
- List any unresolved ambiguities or technical dependencies that require stakeholder clarification.
"""

PROMPT_GAP_ANALYSIS = """
# ROLE & IDENTITY
You are a Strategic Business Analyst and Enterprise Architect AI specializing in Business Process Re-engineering (BPR) and Gap Analysis. Your purpose is to evaluate the distance between an organization's "As-Is" state and "To-Be" state, identify operational, technical, and process gaps, and recommend actionable remediation paths.

# OBJECTIVE
Perform a comprehensive Gap Analysis comparing current processes, systems, and capabilities against future goals. Identify risks, root causes, missing components, and quick-win opportunities.

# CORE ANALYTICAL FRAMEWORK
Evaluate gaps across 4 dimensions:
1. People & Roles (Skills, access levels, handoffs, training)
2. Process & Workflows (Redundancies, bottlenecks, manual steps)
3. Technology & Data (Systems, APIs, data fields, integration gaps)
4. Policy & Compliance (Security, audit trails, regulatory needs)

# OUTPUT FORMATTING RULES
Generate your response strictly using this Markdown structure:

## 1. Executive Gap Summary
High-level narrative outlining the primary delta between current operations and future requirements.

## 2. As-Is vs. To-Be Capability Comparison
| Capability / Process | As-Is State (Current) | To-Be State (Target) | Gap Description | Gap Type (Process/Tech/People/Data) |
| :--- | :--- | :--- | :--- | :--- |
| Onboarding | Manual process | Automated self-service portal | Lack of e-signature integration | Tech / Process |

## 3. Prioritized Gap Assessment & Remediation
For each major identified gap:
- **Gap ID:** GAP-001
- **Description:** What is missing or inadequate?
- **Business Impact:** High / Medium / Low (with explanation of operational impact)
- **Closure Options:** Build vs. Buy vs. Partner vs. Process Change
- **Recommended Remediation:** Actionable steps to bridge the gap

## 4. Prioritization Matrix & Quick Wins
- **Quick Wins (High Impact, Low Effort):** Actions that can be executed immediately.
- **Strategic Initiatives (High Impact, High Effort):** Long-term core structural changes.
- **Low Priority / Backlog (Low Impact):** Non-essential nice-to-haves.
"""

PROMPT_VALIDATION = """
# ROLE & IDENTITY
You are a Lead Quality & Business Analysis Auditor AI specializing in Agile requirements validation, edge-case identification, and acceptance test engineering. Your role is to critically review draft user stories, acceptance criteria, and feature specs to ensure they are complete, unambiguous, and fully testable.

# OBJECTIVE
Audit submitted user stories against the INVEST framework (Independent, Negotiable, Valuable, Estimable, Small, Testable), identify hidden functional/non-functional edge cases, uncover unstated assumptions, and enhance acceptance criteria to prevent production bugs and scope creep.

# OUTPUT FORMATTING RULES
Structure your audit report strictly using this Markdown template:

## 1. Quality Scorecard & Overall Assessment
- **Validation Status:** [APPROVED / APPROVED WITH MODIFICATIONS / NEEDS REVISION]
- **INVEST Score:** X / 10
- **Summary Verdict:** 2-3 sentences highlighting core strengths and critical gaps found.

## 2. Uncovered Edge Cases & Risks
| Risk / Edge Case Category | Identified Scenario | Missing Handling in Draft | Impact (High/Med/Low) |
| :--- | :--- | :--- | :--- |
| Boundary Condition | Inputting zero or negative quantities | No input validation specified | High |

## 3. Ambiguity & Testability Red Flags
*Point out specific words or statements that are untestable or vague:*
- **Flag 1:** "[Quote vague phrase]" -> **Issue:** Cannot be tested objectively. -> **Fix:** Replace with measurable criteria.

## 4. Refined & Production-Ready User Story
- **User Story:** As a [Role], I want [Action] so that [Value].
- **Enhanced Acceptance Criteria (Gherkin Format):**
  - **Scenario 1: Happy Path**
    - **Given** [Context]
    - **When** [Action]
    - **Then** [Expected outcome]
  - **Scenario 2: Edge Case / Exception Handling**
    - **Given** [Context]
    - **When** [Action]
    - **Then** [Expected outcome]

## 5. Recommended Test Scenarios for QA
- List 3-5 specific test cases (functional, security, and performance) that the QA team should execute.
"""


def generate_ba_analysis(provider: str, api_key: str, model_name: str, system_prompt: str, user_input: str) -> str:
    """Executes call to either Google Gemini API or OpenAI API based on user configuration."""
    if provider == "Google Gemini":
        if not HAS_GOOGLE_GENAI:
            raise ImportError("The `google-genai` library is missing. Install it via `pip install google-genai`.")
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )
        return response.text

    elif provider == "OpenAI":
        if not HAS_OPENAI:
            raise ImportError("The `openai` library is missing. Install it via `pip install openai`.")
        
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    else:
        raise ValueError("Unsupported provider selected.")


def main():
    st.title("📊 Business Analyst AI Agent Workbench")
    st.markdown("Automate requirement gathering, conduct gap analysis, and audit user stories for edge cases & testability.")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Agent & API Settings")
        
        # Agent Selection
        agent_type = st.selectbox(
            "Select BA Agent Core Task:",
            [
                "1. Requirement Gathering Agent",
                "2. Gap Analysis Agent",
                "3. Requirements Validation Agent"
            ]
        )
        
        st.divider()
        st.subheader("🤖 LLM Provider Config")
        
        provider = st.radio("Select Provider:", ["Groq (Llama 3.3)", "OpenAI"])
    if provider == "Groq (Llama 3.3)":
    from openai import OpenAI
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
    model_name = "llama-3.3-70b-versatile"

        st.divider()
        st.markdown("---")
        st.caption("💡 **Tip:** Keep temperatures low (0.2) for predictable, structured BA documents.")

    # Agent Information & Dynamic Setup based on selection
    if "Requirement Gathering" in agent_type:
        active_prompt = PROMPT_REQUIREMENT_GATHERING
        agent_desc = "Converts loose client notes, transcripts, or feature ideas into structured BRDs, RTM matrices, and Agile User Stories."
        placeholder_text = "Example: We need a feature in our e-commerce app where customers can schedule recurring monthly deliveries for pet food. They should be able to pause or cancel anytime, pick a delivery date, and get 5% off."
    elif "Gap Analysis" in agent_type:
        active_prompt = PROMPT_GAP_ANALYSIS
        agent_desc = "Compares current 'As-Is' operational states against target 'To-Be' specs to highlight missing data fields, tech gaps, and risks."
        placeholder_text = "As-Is: Currently, customer refund requests are submitted via email. Customer support manually checks the database, processes payment through Stripe dashboard, and emails back within 48h.\n\nTo-Be: Self-service refund button in the customer account portal. Instant automated validation against 14-day return policy and automatic Stripe API refund trigger."
    else:
        active_prompt = PROMPT_VALIDATION
        agent_desc = "Audits draft user stories against the INVEST framework, uncovers missing edge cases, and sharpens acceptance criteria."
        placeholder_text = "Draft Story: As a user, I want to search for products quickly so I can buy them.\nAcceptance Criteria: User enters keyword and sees results fast."

    # Top Agent Card Info
    st.info(f"**Selected Agent:** {agent_type.split('.')[1].strip()}\n\n{agent_desc}")

    # Main Input Area
    user_input = st.text_area("Enter Business Context, Raw Requirements, or Draft Stories:", height=220, placeholder=placeholder_text)

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run BA Agent", type="primary", use_container_width=True)

    if run_btn:
        if not api_key:
            st.error("Please provide a valid API Key in the sidebar to proceed.")
            return
        if not user_input.strip():
            st.warning("Please enter business requirements or context to analyze.")
            return

        with st.spinner("Analyzing requirements and generating BA artifacts..."):
            try:
                output_markdown = generate_ba_analysis(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    system_prompt=active_prompt,
                    user_input=user_input
                )
                st.session_state["ba_output"] = output_markdown
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error executing request: {str(e)}")
                return

    # Display tabbed output if analysis exists in session state
    if "ba_output" in st.session_state:
        st.divider()
        st.subheader("📋 Generated Business Analysis Deliverables")

        tab_raw, tab_matrix, tab_stories, tab_audit = st.tabs([
            "📄 Full Artifact (Markdown)",
            "📊 Traceability / Gap Matrix",
            "🧩 User Stories & Specs",
            "🛡️ Edge Cases & Audit Scorecard"
        ])

        output = st.session_state["ba_output"]

        with tab_raw:
            st.markdown(output)
            st.download_button(
                label="📥 Download Artifact (.md)",
                data=output,
                file_name="BA_Requirements_Deliverable.md",
                mime="text/markdown"
            )

        with tab_matrix:
            st.markdown("### Matrix View")
            # Filter and show tables extracted from markdown
            tables = re.findall(r"(\|(?:[^\n]+\|\n)+)", output)
            if tables:
                for idx, table in enumerate(tables):
                    st.markdown(table)
            else:
                st.info("No specific matrix tables detected in output. See Full Artifact tab.")

        with tab_stories:
            st.markdown("### Agile User Stories & Acceptance Criteria")
            if "User Story" in output or "User Stories" in output:
                st.markdown(output)
            else:
                st.info("User stories view is optimized for Requirement Gathering & Validation agents.")

        with tab_audit:
            st.markdown("### Edge Cases, Scorecard & Risks")
            if "Scorecard" in output or "Edge Cases" in output or "Gap" in output:
                st.markdown(output)
            else:
                st.info("Audit view is optimized for Gap Analysis & Validation agents.")


if __name__ == "__main__":
    main()
