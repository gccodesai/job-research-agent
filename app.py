"""Web interface. Run locally with:  streamlit run app.py"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import NotFoundError

from agent import PROVIDERS, make_client, model_for, run_agent

load_dotenv()

st.set_page_config(page_title="Job Posting Research Agent", page_icon="🔎", layout="wide")


def get_secret(name: str) -> str | None:
    """Read a key from Streamlit Cloud secrets, falling back to .env."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file locally
        pass
    return os.getenv(name)


def read_resume_file(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(uploaded).pages)
    return uploaded.read().decode("utf-8", errors="ignore")


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    provider_id = st.selectbox(
        "LLM provider (both free)",
        options=list(PROVIDERS),
        format_func=lambda p: PROVIDERS[p].name,
    )
    provider = PROVIDERS[provider_id]
    saved_key = get_secret(provider.key_env_var)
    api_key = st.text_input(
        f"{provider.name} API key",
        type="password",
        placeholder="Using saved key" if saved_key else "Paste your key",
        help=f"Free key, no credit card: {provider.signup_url}",
    ) or saved_key
    model = st.text_input(
        "Model",
        value=model_for(provider_id),
        help="Change this if the provider retires a model (you'll see a 404 error).",
    )
    max_steps = st.slider("Max research steps", 3, 12, 8)
    st.caption(
        "⚠️ Free tiers may use your inputs for training. Remove your phone "
        "number and address from your resume before pasting it."
    )

# ---------- Main: inputs ----------
st.title("🔎 Job Posting Research Agent")
st.write(
    "Give it a job posting and your resume. The agent researches the company, "
    "compares the role against your experience, and writes an interview prep report."
)

col1, col2 = st.columns(2)
with col1:
    job_posting = st.text_area(
        "Job posting: paste the text, or a URL",
        height=300,
        placeholder="https://careers.example.com/jobs/123  — or paste the full posting",
    )
    st.caption("Tip: LinkedIn and Indeed block automated access, so paste the text for those.")
with col2:
    uploaded = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt", "md"])
    resume_text = st.text_area("…or paste your resume", height=220)

if st.button("Research this job", type="primary", use_container_width=True):
    resume = read_resume_file(uploaded) if uploaded else resume_text
    if not job_posting.strip() or not resume.strip():
        st.error("Please provide both a job posting and a resume.")
        st.stop()
    if not api_key:
        st.error(f"Add a {provider.name} API key in the sidebar.")
        st.stop()

    client = make_client(provider_id, api_key)
    with st.status("Agent is researching…", expanded=True) as status:

        def show_step(step):
            icon = "🌐" if step.tool == "web_search" else "📄"
            st.write(f"{icon} **{step.tool}** `{step.arguments}`")

        try:
            result = run_agent(
                job_posting, resume, client, model,
                max_steps=max_steps, on_step=show_step,
            )
        except NotFoundError:
            status.update(label="Model not found", state="error")
            st.error(
                f"{provider.name} doesn't recognise the model '{model}'. It may have "
                f"been retired. Change the Model box in the sidebar to a current model "
                f"name from the provider's docs and try again."
            )
            st.stop()
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()
        status.update(
            label=f"Done: {len(result.steps)} research steps", state="complete", expanded=False
        )

    if result.hit_step_limit:
        st.info("The agent hit the step limit, so the report uses what it found so far.")
    st.markdown(result.report)
    st.download_button(
        "Download report (Markdown)", result.report, file_name="job_research_report.md"
    )
