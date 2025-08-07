import streamlit as st
import json
import time
import asyncio
from datetime import datetime
import pytz

# Assuming these are defined/imported elsewhere in your codebase:
from your_project import (
    process_url_workflow_with_logging,
    ContentExtractor,
    ChunkProcessor,
    extract_big_chunks,
    process_ai_analysis,
    create_export_options,
    ANALYZER_ASSISTANT_ID,
    logger,
)

# --- Streamlit UI ---
def main():
    st.title("🕵 YMYL Audit Tool")
    st.markdown("**Automatically extract content from websites, generate JSON chunks, and perform YMYL compliance analysis**")

    # Sidebar configuration
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=True, help="Show detailed processing logs")

    # API Key configuration
    st.sidebar.markdown("### 🔑 AI Analysis Configuration")
    try:
        api_key = st.secrets["openai_api_key"]
        st.sidebar.success("✅ API Key loaded from secrets")
    except Exception:
        api_key = st.sidebar.text_input(
            "OpenAI API Key:",
            type="password",
            help="Enter your OpenAI API key for AI analysis"
        )
        if api_key:
            st.sidebar.success("✅ API Key provided")
        else:
            st.sidebar.warning("⚠️ API Key needed for AI analysis")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        url = st.text_input("Enter the URL to process:", help="Include http:// or https://")
        if st.button("🚀 Process URL", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a URL to process")
                return

            # clear old state
            for key in ("latest_result", "ai_analysis_result", "chunk_done"):  # also reset chunk lights
                if key in st.session_state:
                    st.session_state.pop(key)

            if debug_mode:
                # Detailed logging
                log_placeholder = st.empty()
                log_lines = []
                def log_callback(msg):
                    now = datetime.now(pytz.timezone("Europe/Malta"))
                    log_lines.append(f"`{now.strftime('%H:%M:%S')}`: {msg}")
                    log_placeholder.info("\n".join(log_lines))

                result = process_url_workflow_with_logging(url, log_callback)
                st.session_state["latest_result"] = result
                if result["success"]:
                    st.success("Processing completed successfully!")
                else:
                    st.error(f"Error: {result['error']}")
            else:
                # Simple milestones
                log_area = st.empty()
                milestones = []
                def simple_log(text):
                    milestones.append(f"- {text}")
                    log_area.markdown("\n".join(milestones))

                simple_log("Extracting content")
                extractor = ContentExtractor()
                ok, content, err = extractor.extract_content(url)
                if not ok:
                    st.error(f"Error: {err}")
                    return

                simple_log("Sending content to Chunk Norris")
                processor = ChunkProcessor()

                with st.status("You are not waiting, Chunk Norris is waiting for you"):
                    ok, json_out, err = processor.process_content(content)

                simple_log("Chunking done!")
                st.success("Chunking done!")

                st.session_state["latest_result"] = {
                    "success": ok,
                    "url": url,
                    "extracted_content": content if ok else None,
                    "json_output": json_out if ok else None,
                    "error": err
                }

    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
1. **Extract**: Extract the content.
2. **Chunk**: Send extracted text to Chunk Norris.
3. **YMYL Analysis**: YMYL audit of the content with AI.
4. **Done**: Output complete report.
""")
        st.info("💡 **New**: AI-powered YMYL compliance analysis available!")

    # Results Display
    if 'latest_result' in st.session_state and st.session_state['latest_result'].get('success'):
        result = st.session_state['latest_result']
        st.markdown("---")
        st.subheader("📊 Results")
        
        # AI Analysis Button (label updated)
        if api_key and st.button("🤖 Start YMYL AI Compliance Analysis", type="secondary", use_container_width=True):
            try:
                # Parse JSON and extract chunks first
                json_data = json.loads(result['json_output'])
                chunks = extract_big_chunks(json_data)
                
                if not chunks:
                    st.error("No chunks found in JSON data")
                    return
                
                # Enhanced Processing Logs Section (simplified)
                st.subheader("🔍 Processing Logs")
                log_container = st.container()
                
                with log_container:
                    st.info("🚀 Starting the analysis")

                    # Initialize per-chunk status lights
                    num_chunks = len(chunks)
                    if 'chunk_done' not in st.session_state or len(st.session_state['chunk_done']) != num_chunks:
                        st.session_state['chunk_done'] = [False] * num_chunks
                
                    # Function to render lights
                    def render_chunk_lights():
                        lights_per_row = 10
                        for row_start in range(0, num_chunks, lights_per_row):
                            icons = ["🟢" if st.session_state['chunk_done'][i] else "⚪" 
                                     for i in range(row_start, min(row_start + lights_per_row, num_chunks))]
                            st.write(" ".join(icons))

                    # Show initial lights
                    render_chunk_lights()

                    # Progress tracking
                    progress_bar = st.progress(0)

                    # Callback to update each chunk's status
                    def chunk_callback(idx):
                        st.session_state['chunk_done'][idx] = True
                        render_chunk_lights()
                        done = sum(st.session_state['chunk_done'])
                        progress_bar.progress(done / num_chunks)

                # Start processing with timing
                start_time = time.time()

                with st.spinner("🤖 Running parallel analysis..."):
                    # Run AI analysis with per-chunk callback
                    success, ai_result, analysis_details = asyncio.run(
                        process_ai_analysis(
                            result['json_output'], 
                            api_key, 
                            chunk_callback  # now receives updates per chunk
                        )
                    )

                processing_time = time.time() - start_time
                # Ensure completion indicator
                progress_bar.progress(1.0)

                # Display processing summary
                if success and analysis_details:
                    successful_analyses = [r for r in analysis_details if r.get("success")]
                    failed_analyses = [r for r in analysis_details if not r.get("success")]
                    
                    status_container = st.empty()
                    with status_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Chunks", num_chunks)
                        with col2:
                            st.metric("Successful", len(successful_analyses), 
                                     delta=(len(successful_analyses) if len(successful_analyses) == num_chunks else None))
                        with col3:
                            st.metric("Failed", len(failed_analyses), 
                                     delta=(f"-{len(failed_analyses)}" if len(failed_analyses) > 0 else None))
                        
                        st.success(f"✅ Parallel analysis completed in {processing_time:.2f} seconds")
                    
                    # Store results
                    st.session_state['ai_analysis_result'] = {
                        'success': True,
                        'report': ai_result,
                        'details': analysis_details,
                        'processing_time': processing_time,
                        'total_chunks': num_chunks,
                        'successful_count': len(successful_analyses),
                        'failed_count': len(failed_analyses)
                    }
                    
                else:
                    st.session_state['ai_analysis_result'] = {
                        'success': False,
                        'error': ai_result if not success else 'Unknown error occurred'
                    }
                    st.error(f"❌ AI analysis failed: {ai_result if not success else 'Unknown error'}")

            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON format: {str(e)}")
            except Exception as e:
                st.error(f"❌ An error occurred during AI analysis: {str(e)}")
                logger.error(f"AI analysis error: {str(e)}")

        # ... rest of your tabs and result display unchanged ...

if __name__ == "__main__":
    main()
