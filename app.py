#!/usr/bin/env python3
"""
Parallel Analysis Test App

Tests the parallel chunk analysis system with real OpenAI Assistants API calls.
"""

import streamlit as st
import json
import asyncio
import aiohttp
import time
from datetime import datetime
import logging
from typing import List, Dict, Any

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Parallel Analysis Test",
    page_icon="🧪",
    layout="wide",
)

# --- Assistant IDs ---
ANALYZER_ASSISTANT_ID = "asst_WzODK9EapCaZoYkshT6x9xEH"

# --- Report Maker Prompt (UPDATED) ---
# This prompt is enhanced to create an executive summary with a grade, table, and priority actions.
REPORT_MAKER_PROMPT = """**Persona:** You are a senior compliance report writer and SEO strategist with 15 years of experience in the YMYL space, specializing in the online casino and gambling industry. You are known for creating executive-ready, actionable reports that are clear, concise, and professional.

**Task:** You will be given a set of 'Issue Cards' from a YMYL compliance audit, along with a list of any sections that failed analysis. Your goal is to synthesize this information into a single, cohesive, and high-impact audit report formatted in clean Markdown.

**CRITICAL INSTRUCTIONS:**

1.  **Start with an Executive Summary:** This is the most important part of the report. It must appear at the very top.
    * **Overall Score:** Begin with an overall "Compliance Grade" (e.g., C+). Briefly explain your scoring logic. For example: "Start with 100 points. Subtract 10 for each Critical issue, 5 for each High, 2 for each Medium, and 1 for each Low."
    * **Summary Table:** Create a Markdown table that summarizes the findings by counting the severity emojis (🔴, 🟠, 🟡, 🔵).
    * **Priority Actions:** List the top 3-5 most critical issues as a bulleted list. This should be a direct call to action.

2.  **Create the Detailed Findings Section:**
    * After the Executive Summary, add a main heading: `## Detailed Findings`.
    * Organize the provided 'Issue Cards' under the content section they came from, using the section's descriptive name (e.g., `### Analysis for Section 1: Welcome Bonus`).
    * **Preserve the exact formatting** of each 'Issue Card'. Do not alter or rephrase them.

3.  **Add a Concluding Summary:**
    * End the report with `## Processing Summary`.
    * Note how many sections were analyzed successfully and list any sections that failed analysis based on the input provided.

**Final Output Requirements:**
* The entire output must be a single block of well-formatted Markdown.
* Use headers, bold text, lists, and tables to ensure the report is professional and scannable.
* The report date should be {current_date}.
"""

# --- OpenAI Assistant API Functions (UNCHANGED) ---
async def call_assistant(api_key: str, assistant_id: str, content: str, chunk_index: int = None) -> Dict[str, Any]:
    """Make async call to OpenAI Assistant API"""
    try:
        logger.info(f"Starting assistant call for chunk {chunk_index} with assistant {assistant_id}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }

        # Create thread
        logger.info("Creating thread...")
        thread_response = await create_thread(headers)
        if not thread_response["success"]:
            logger.error(f"Thread creation failed: {thread_response['error']}")
            return {"success": False, "error": thread_response["error"], "chunk_index": chunk_index}

        thread_id = thread_response["thread_id"]
        logger.info(f"Thread created: {thread_id}")

        # Add message to thread
        logger.info("Adding message to thread...")
        message_response = await add_message_to_thread(headers, thread_id, content)
        if not message_response["success"]:
            logger.error(f"Message creation failed: {message_response['error']}")
            return {"success": False, "error": message_response["error"], "chunk_index": chunk_index}

        # Run assistant
        logger.info("Running assistant...")
        run_response = await run_assistant(headers, thread_id, assistant_id)
        if not run_response["success"]:
            logger.error(f"Assistant run failed: {run_response['error']}")
            return {"success": False, "error": run_response["error"], "chunk_index": chunk_index}

        logger.info(f"Assistant call completed successfully for chunk {chunk_index}")
        return {
            "success": True,
            "content": run_response["content"],
            "chunk_index": chunk_index,
            "tokens_used": run_response.get("tokens_used", 0)
        }

    except Exception as e:
        logger.error(f"Exception in call_assistant: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "chunk_index": chunk_index
        }

async def create_thread(headers: dict) -> Dict[str, Any]:
    """Create a new thread"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/threads",
                headers=headers,
                json={},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "thread_id": result["id"]}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Thread creation failed: {error_text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def add_message_to_thread(headers: dict, thread_id: str, content: str) -> Dict[str, Any]:
    """Add message to thread"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.openai.com/v1/threads/{thread_id}/messages",
                headers=headers,
                json={
                    "role": "user",
                    "content": content
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return {"success": True}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Message creation failed: {error_text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def run_assistant(headers: dict, thread_id: str, assistant_id: str) -> Dict[str, Any]:
    """Run assistant and poll for completion"""
    try:
        # Create run
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.openai.com/v1/threads/{thread_id}/runs",
                headers=headers,
                json={
                    "assistant_id": assistant_id
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"Run creation failed: {error_text}"}

                run_result = await response.json()
                run_id = run_result["id"]

        # Poll for completion
        max_attempts = 60  # 5 minutes max
        attempt = 0

        while attempt < max_attempts:
            await asyncio.sleep(5)  # Wait 5 seconds between polls

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        run_status = await response.json()
                        status = run_status["status"]

                        logger.info(f"Run status: {status}")

                        if status == "completed":
                            # Get messages
                            return await get_thread_messages(headers, thread_id)
                        elif status in ["failed", "cancelled", "expired"]:
                            return {"success": False, "error": f"Run {status}"}
                        # Continue polling for other statuses
                    else:
                        # If polling fails, just continue to the next attempt
                        pass

            attempt += 1

        return {"success": False, "error": "Run polling timeout"}

    except Exception as e:
        return {"success": False, "error": str(e)}

async def get_thread_messages(headers: dict, thread_id: str) -> Dict[str, Any]:
    """Get messages from thread"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.openai.com/v1/threads/{thread_id}/messages",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    messages = await response.json()
                    # Get the latest assistant message
                    for message in messages["data"]:
                        if message["role"] == "assistant":
                            content = message["content"][0]["text"]["value"]
                            return {"success": True, "content": content}

                    return {"success": False, "error": "No assistant message found"}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Message retrieval failed: {error_text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Core Logic Functions (UNCHANGED and UPDATED) ---
def extract_big_chunks(json_data: Dict) -> List[Dict]:
    """Extract and prepare big chunks for analysis"""
    chunks = []

    if "big_chunks" not in json_data:
        return chunks

    for big_chunk in json_data["big_chunks"]:
        chunk_index = big_chunk.get("big_chunk_index", 0)
        small_chunks = big_chunk.get("small_chunks", [])

        # Join small chunks with newlines (Option A)
        chunk_text = "\n".join(small_chunks)

        chunks.append({
            "index": chunk_index,
            "text": chunk_text,
            "small_chunks_count": len(small_chunks)
        })

    return chunks

async def process_chunks_parallel(chunks: List[Dict], api_key: str) -> List[Dict]:
    """Process all chunks in parallel using analyzer assistant"""
    tasks = []

    for chunk in chunks:
        task = call_assistant(
            api_key=api_key,
            assistant_id=ANALYZER_ASSISTANT_ID,
            content=chunk["text"],
            chunk_index=chunk["index"]
        )
        tasks.append(task)

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "error": str(result),
                "chunk_index": chunks[i]["index"]
            })
        else:
            processed_results.append(result)

    return processed_results

async def generate_report_with_chat(api_key: str, report_input: str) -> Dict[str, Any]:
    """Generate report using fast Chat Completions API"""
    try:
        logger.info("Starting report generation with Chat Completions")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Format the prompt with current date
        formatted_prompt = REPORT_MAKER_PROMPT.format(current_date=datetime.now().strftime("%Y-%m-%d"))

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": report_input}
            ],
            "max_tokens": 4000,
            "temperature": 0.1
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120) # Increased timeout for larger reports
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logger.info("Report generated successfully with Chat Completions")
                    return {
                        "success": True,
                        "content": result["choices"][0]["message"]["content"],
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Chat completions failed: {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }

    except Exception as e:
        logger.error(f"Exception in generate_report_with_chat: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_report_input(analysis_results: List[Dict], chunks: List[Dict]) -> str:
    """(UPDATED) Prepares the structured input for the new Report Maker AI."""
    
    successful_analyses = []
    failed_sections = []

    # Create a mapping of chunk index to its name for easy lookup
    chunk_names = {
        chunk['index']: next(
            (s.split(":", 1)[1].strip() for s in chunk['text'].split('\n') if s.startswith("H1:")), 
            f"Section {chunk['index']}"
        ) for chunk in chunks
    }

    for result in analysis_results:
        chunk_index = result["chunk_index"]
        section_name = chunk_names.get(chunk_index, f"Section {chunk_index}")

        if result.get("success") and "No issues found" not in result.get("content", ""):
            # Add section name context to the analysis for the report maker
            analysis_with_context = f"### Analysis for {section_name}\n\n{result['content']}"
            successful_analyses.append(analysis_with_context)
        elif not result.get("success"):
            failed_sections.append(f"- **{section_name}**: {result.get('error', 'Unknown error')}")

    # Build the final input string for the AI
    report_input_parts = []
    if successful_analyses:
        report_input_parts.append("--- DETAILED ANALYSES ---\n" + "\n\n".join(successful_analyses))
    else:
        report_input_parts.append("--- DETAILED ANALYSES ---\nNo issues were found in any of the successfully analyzed sections.")

    if failed_sections:
        report_input_parts.append("\n\n--- FAILED SECTIONS ---\n" + "\n".join(failed_sections))

    return "\n\n".join(report_input_parts)


# --- Streamlit UI (UNCHANGED) ---
def main():
    st.title("🧪 Parallel Analysis Test App")
    st.markdown("**Test the parallel chunk analysis system with real OpenAI Assistants API calls**")

    # Check for API key
    if "openai_api_key" not in st.secrets:
        st.error("❌ API key not configured. Please add `openai_api_key` to your secrets.")
        st.stop()

    st.success("✅ API key loaded from secrets")
    st.markdown("---")

    # Input section
    st.subheader("📥 Input JSON Data")
    st.markdown("Paste the JSON output from chunk.dejan.ai:")

    json_input = st.text_area(
        "JSON Content:",
        height=300,
        placeholder='{\n  "big_chunks": [\n    {\n      "big_chunk_index": 1,\n      "small_chunks": ["content..."]\n    }\n  ]\n}'
    )

    if st.button("🚀 Start Parallel Analysis", type="primary"):
        if not json_input.strip():
            st.error("Please provide JSON input")
            return

        try:
            # Parse JSON
            json_data = json.loads(json_input)

            # Extract chunks
            chunks = extract_big_chunks(json_data)

            if not chunks:
                st.error("No big_chunks found in JSON data")
                return

            st.success(f"✅ Found {len(chunks)} big chunks to analyze")

            # Show chunks
            with st.expander("📋 Extracted Chunks Preview"):
                for chunk in chunks:
                    st.write(f"**Chunk {chunk['index']}** ({chunk['small_chunks_count']} small chunks)")
                    st.text(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
                    st.markdown("---")

            # Add detailed logging
            st.subheader("🔍 Processing Logs")
            log_container = st.container()

            api_key = st.secrets["openai_api_key"]

            with log_container:
                st.info(f"🚀 Starting parallel analysis of {len(chunks)} chunks...")
                st.write(f"**Assistant IDs:**")
                st.write(f"- Analyzer: `{ANALYZER_ASSISTANT_ID}`")
                st.write(f"- Report Maker: `Chat Completions (gpt-4o)`")
                st.write(f"**API Key Status:** {'✅ Loaded' if api_key.startswith('sk-') else '❌ Invalid'}")

            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.empty()

            # Start processing
            start_time = time.time()

            try:
                with st.spinner("🤖 Running parallel analysis... This may take several minutes."):
                    async def run_analysis():
                        logger.info(f"Starting analysis of {len(chunks)} chunks")
                        return await process_chunks_parallel(chunks, api_key)

                    # Execute async code
                    analysis_results = asyncio.run(run_analysis())

                logger.info("Analysis completed")

                # Update progress
                progress_bar.progress(1.0)
                processing_time = time.time() - start_time

                # Display results
                successful_analyses = [r for r in analysis_results if r["success"]]
                failed_analyses = [r for r in analysis_results if not r["success"]]

                with status_container.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", len(chunks))
                    with col2:
                        st.metric("Successful", len(successful_analyses))
                    with col3:
                        st.metric("Failed", len(failed_analyses))

                    st.success(f"✅ Parallel analysis completed in {processing_time:.2f} seconds")

                # Show individual results
                st.subheader("📊 Individual Chunk Analyses")

                for result in sorted(analysis_results, key=lambda x: x['chunk_index']):
                    chunk_idx = result["chunk_index"]

                    if result["success"]:
                        with st.expander(f"✅ Chunk {chunk_idx} Analysis (Success)"):
                            st.markdown(result["content"])
                            st.caption(f"Tokens used: {result.get('tokens_used', 'Unknown')}")
                    else:
                        with st.expander(f"❌ Chunk {chunk_idx} Analysis (Failed)"):
                            st.error(f"Error: {result['error']}")

                # Generate final report
                if successful_analyses or failed_analyses:
                    st.subheader("📋 Final Unified Report")

                    with st.spinner("📝 Generating unified report..."):
                        try:
                            # The call here is now UPDATED to pass `chunks`
                            report_input = create_report_input(analysis_results, chunks)
                            logger.info(f"Report input created: {len(report_input)} characters")

                            # Call report maker using fast Chat Completions
                            async def generate_report():
                                logger.info("Starting fast report generation")
                                return await generate_report_with_chat(api_key, report_input)

                            report_result = asyncio.run(generate_report())

                            if report_result["success"]:
                                st.success("✅ Report generated successfully!")

                                # Use Streamlit's built-in copy functionality
                                st.markdown("**Click the copy button (📋) in the top-right corner of the code block below:**")

                                # Display report in code block with copy button
                                st.code(report_result["content"], language="markdown")

                                # Also show formatted preview
                                with st.expander("📖 View Formatted Report Preview", expanded=True):
                                    st.markdown(report_result["content"])

                                logger.info("Report generated successfully")
                            else:
                                st.error(f"❌ Failed to generate report: {report_result['error']}")
                                logger.error(f"Report generation failed: {report_result['error']}")

                        except Exception as e:
                            st.error(f"❌ Error generating report: {str(e)}")
                            logger.error(f"Report generation error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                logger.error(f"Analysis error: {str(e)}")

        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
            logger.error(f"JSON decode error: {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logger.error(f"General error: {str(e)}")

if __name__ == "__main__":
    main()
