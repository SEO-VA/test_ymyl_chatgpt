import streamlit as st
import openai
import json
import asyncio
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="YMYL Parallel Audit System",
    page_icon="🔬",
    layout="wide"
)

# --- App Title and Description ---
st.title("🔬 YMYL Parallel Content Audit System")
st.markdown("""
This application processes structured content from `chunk.dejan.ai` for YMYL compliance.
1.  Paste the JSON output from `chunk.dejan.ai` into the text area below.
2.  Enter your Analyzer Assistant ID.
3.  Click "Run Parallel Analysis" to begin the audit.
""")

# --- OpenAI Client Initialization ---
# Best practice: Use Streamlit Secrets for API key management
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI API key loaded from secrets.", icon="✅")
except KeyError:
    st.sidebar.error("ERROR: `OPENAI_API_KEY` not found in Streamlit Secrets.")
    st.stop()


# --- Core Functions ---

def get_report_maker_prompt():
    """Returns the full text of the advanced Report Maker prompt."""
    return """
**Persona:** You are a senior compliance report writer and SEO strategist with 15 years of experience in the YMYL space, specializing in the online casino and gambling industry. You are known for creating executive-ready, actionable reports that are clear, concise, and professional.

**Task:** You will be given a list of strings, where each string is a pre-formatted 'Issue Card' in Markdown from a YMYL compliance audit. Some sections might be marked as having failed analysis. Your goal is to synthesize these individual analyses into a single, cohesive, and high-impact audit report formatted in clean Markdown.

**CRITICAL INSTRUCTIONS:**

1.  **Start with an Executive Summary:** This is the most important part of the report. It must appear at the very top.
    * **Overall Score:** Begin with an overall "Compliance Grade" (e.g., C+). Briefly explain your scoring logic. For example: "Start with 100 points. Subtract 10 for each Critical issue, 5 for each High, 2 for each Medium, and 1 for each Low."
    * **Summary Table:** Create a Markdown table that summarizes the findings by severity.
    * **Priority Actions:** List the top 3-5 most critical issues as a bulleted list. This should be a direct call to action.

2.  **Create the Detailed Findings Section:**
    * After the Executive Summary, add a main heading: `## Detailed Findings`.
    * Organize the provided 'Issue Cards' under the content section they came from, using the section's descriptive name.
    * **Preserve the exact formatting** of each 'Issue Card'. Do not alter or rephrase them.

3.  **Add a Concluding Summary:**
    * End the report with `## Processing Summary`.
    * Note how many sections were analyzed successfully and which failed (if any).

**Final Output Requirements:**
* The entire output must be a single block of well-formatted Markdown.
* Use headers, bold text, lists, and tables to ensure the report is professional and scannable.
"""

async def analyze_chunk(client, assistant_id, section_name, content, progress_bar, status_log):
    """
    Asynchronously analyzes a single content chunk using the OpenAI Assistants API.
    """
    try:
        status_log.text(f"  - ⏳ Starting analysis for: {section_name}")
        
        # 1. Create a Thread
        thread = await client.beta.threads.create()

        # 2. Add a Message to the Thread
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content
        )

        # 3. Run the Assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # 4. Poll for completion
        while run.status not in ["completed", "failed"]:
            await asyncio.sleep(2) # Non-blocking sleep
            run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            status_log.text(f"  - ⚙️ Processing: {section_name} (Status: {run.status})")

        if run.status == "failed":
             raise Exception(f"Run failed. Reason: {run.last_error.message}")

        # 5. Get the Assistant's Response
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        response_content = messages.data[0].content[0].text.value
        
        progress_bar.progress(1, text=f"✅ Completed: {section_name}")
        status_log.text(f"  - ✅ Completed analysis for: {section_name}")
        return {"section": section_name, "analysis": response_content, "status": "success"}

    except Exception as e:
        error_message = f"Error processing {section_name}: {str(e)}"
        st.error(error_message, icon="🚨")
        status_log.text(f"  - 🚨 FAILED: {section_name}")
        return {"section": section_name, "analysis": error_message, "status": "error"}


# --- Streamlit UI ---

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    # It's better to keep the assistant_id in the main app for testability
    # The default value is the one from the project description
    analyzer_assistant_id = st.text_input(
        "Analyzer Assistant ID",
        value="asst_WzODK9EapCaZoYkshT6x9xEH",
        help="The ID of the OpenAI Assistant configured with the Analyzer prompt."
    )

# Main content area
json_input = st.text_area(
    "Paste JSON from chunk.dejan.ai here",
    height=250,
    placeholder='{"big_chunks":[{"big_chunk_index":1,"small_chunks":["H1: Your Title..."]}]}'
)

if st.button("🚀 Run Parallel Analysis", type="primary"):
    if not json_input:
        st.warning("Please paste the JSON content first.")
        st.stop()
    if not analyzer_assistant_id:
        st.warning("Please enter your Analyzer Assistant ID.")
        st.stop()

    try:
        data = json.loads(json_input)
        big_chunks = data["big_chunks"]
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Invalid JSON format. Please check the input. Error: {e}")
        st.stop()

    st.success(f"JSON parsed successfully. Found {len(big_chunks)} sections to analyze.")

    # --- Asynchronous Processing Logic ---
    async def main():
        # Use the async-compatible OpenAI client
        async_client = openai.AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # UI placeholders
        progress_text = "Analysis in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        status_log_placeholder = st.empty()
        
        with st.expander("🔬 Live Analysis Log", expanded=True):
            status_log = st.empty()

        tasks = []
        for i, chunk in enumerate(big_chunks):
            # Reconstruct content from small_chunks, preserving formatting
            content_text = "\n".join(chunk["small_chunks"])
            # Create a descriptive section name (e.g., from the H1)
            section_name = f"Section {i+1}: " + next((s.split(":", 1)[1].strip() for s in chunk["small_chunks"] if s.startswith("H1:")), "Untitled")
            
            # Add task to the list for parallel execution
            tasks.append(analyze_chunk(async_client, analyzer_assistant_id, section_name, content_text, my_bar, status_log))
        
        # Run all analysis tasks concurrently
        analysis_results = await asyncio.gather(*tasks)

        my_bar.progress(1.0, text="All sections processed!")
        status_log.text("All analyses complete. Generating final report...")

        # --- Report Generation ---
        st.subheader("📊 Individual Section Analyses")
        successful_analyses = []
        for result in analysis_results:
            with st.expander(f"{'✅' if result['status'] == 'success' else '🚨'} {result['section']}"):
                st.markdown(result['analysis'])
            if result['status'] == 'success' and "No issues found" not in result['analysis']:
                # Add section name context to the analysis for the report maker
                analysis_with_context = f"### Analysis for {result['section']}\n\n" + result['analysis']
                successful_analyses.append(analysis_with_context)

        # Combine results for the Report Maker
        if successful_analyses:
            combined_analysis_text = "\n\n".join(successful_analyses)
            
            # Call Chat Completions API for the final report
            report_maker_prompt = get_report_maker_prompt()
            
            report_response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": report_maker_prompt},
                    {"role": "user", "content": combined_analysis_text}
                ]
            )
            final_report = report_response.choices[0].message.content

            st.subheader("📋 Final Compliance Report")
            with st.expander("Click to view and copy the full report", expanded=True):
                st.code(final_report, language='markdown')

        else:
            st.warning("No successful analyses were completed or no issues were found. Final report cannot be generated.")


    # Run the main async function
    asyncio.run(main())
