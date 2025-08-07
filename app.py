#!/usr/bin/env python3
"""
Content Processing Web Application with AI Compliance Analysis

This script combines the robust content extraction and chunking with
parallel AI-powered YMYL compliance analysis capabilities.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import json
import time
import html
from datetime import datetime
import pytz
import platform
import logging
import asyncio
import aiohttp
from openai import OpenAI

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Content Processor with AI Analysis",
    page_icon="🚀",
    layout="wide",
)

# --- AI Processing Configuration ---
ANALYZER_ASSISTANT_ID = "asst_WzODK9EapCaZoYkshT6x9xEH"

# --- Component 1: Updated Content Extractor ---
class ContentExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_content(self, url):
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_parts = []
            
            # 1. Extract H1 (anywhere on page)
            h1 = soup.find('h1')
            if h1:
                text = h1.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(f"H1: {text}")
            
            # 2. Extract Subtitle (anywhere on page)
            subtitle = soup.find('span', class_=['sub-title', 'd-block'])
            if subtitle:
                text = subtitle.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(f"SUBTITLE: {text}")
            
            # 3. Extract Lead (anywhere on page)
            lead = soup.find('p', class_='lead')
            if lead:
                text = lead.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(f"LEAD: {text}")
            
            # 4. Extract Article content
            article = soup.find('article')
            if article:
                # Remove tab-content sections before processing
                for tab_content in article.find_all('div', class_='tab-content'):
                    tab_content.decompose()
                
                # Process all elements in document order within article
                for element in article.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'p']):
                    text = element.get_text(separator='\n', strip=True)
                    if not text:
                        continue
                    
                    # Check element type and add appropriate prefix
                    if element.name == 'h1':
                        content_parts.append(f"H1: {text}")
                    elif element.name == 'h2':
                        content_parts.append(f"H2: {text}")
                    elif element.name == 'h3':
                        content_parts.append(f"H3: {text}")
                    elif element.name == 'h4':
                        content_parts.append(f"H4: {text}")
                    elif element.name == 'h5':
                        content_parts.append(f"H5: {text}")
                    elif element.name == 'h6':
                        content_parts.append(f"H6: {text}")
                    elif element.name == 'span' and 'sub-title' in element.get('class', []) and 'd-block' in element.get('class', []):
                        content_parts.append(f"SUBTITLE: {text}")
                    elif element.name == 'p' and 'lead' in element.get('class', []):
                        content_parts.append(f"LEAD: {text}")
                    elif element.name == 'p':
                        content_parts.append(f"CONTENT: {text}")
            
            # 5. Extract FAQ section
            faq_section = soup.find('section', attrs={'data-qa': 'templateFAQ'})
            if faq_section:
                text = faq_section.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(f"FAQ: {text}")
            
            # 6. Extract Author section
            author_section = soup.find('section', attrs={'data-qa': 'templateAuthorCard'})
            if author_section:
                text = author_section.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(f"AUTHOR: {text}")
            
            # Join with double newlines to preserve spacing
            final_content = '\n\n'.join(content_parts)
            return True, final_content, None
            
        except requests.RequestException as e:
            return False, None, f"Error fetching URL: {e}"
        except Exception as e:
            return False, None, f"Error processing content: {e}"

# --- Component 2: The Final, Upgraded Chunk Processor ---
class ChunkProcessor:
    def __init__(self, log_callback=None):
        self.driver = None
        self.log = log_callback if log_callback else logger.info

    def _setup_driver(self):
        self.log("Initializing browser with enhanced stability & permissions...")
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.clipboard": 1})
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.log("✅ Browser initialized successfully.")
            return True
        except WebDriverException as e:
            self.log(f"❌ WebDriver Initialization Failed: {e}")
            return False

    def _extract_json_from_button(self):
        try:
            wait = WebDriverWait(self.driver, 180)
            h3_xpath = "//h3[text()='Raw JSON Output']"
            self.log("🔄 Waiting for results section to appear...")
            wait.until(EC.presence_of_element_located((By.XPATH, h3_xpath)))
            self.log("✅ Results section is visible.")
            button_selector = "button[data-testid='stCodeCopyButton']"
            self.log("...Waiting for the copy button...")
            copy_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, button_selector)))
            self.log("✅ Found the copy button element.")
            self.log("...Polling button's attribute for completeness...")
            timeout = time.time() + 10
            final_content = ""
            while time.time() < timeout:
                raw_content = copy_button.get_attribute('data-clipboard-text')
                if raw_content and raw_content.strip().startswith('{') and raw_content.strip().endswith('}'):
                    final_content = raw_content; break
                time.sleep(0.2)
            if not final_content: self.log("❌ Timed out polling the attribute."); return None
            self.log("...Decoding HTML entities...")
            decoded_content = html.unescape(final_content)
            self.log(f"✅ Extraction complete. Retrieved {len(decoded_content):,} characters.")
            return decoded_content
        except Exception as e:
            self.log(f"❌ An error occurred during the final JSON extraction phase: {e}")
            return None

    def process_content(self, content):
        if not self._setup_driver():
            return False, None, "Failed to initialize browser."
        try:
            self.log(f"Navigating to `chunk.dejan.ai`...")
            self.driver.get("https://chunk.dejan.ai/")
            wait = WebDriverWait(self.driver, 30)
            self.log("Using JavaScript to copy full text to browser's clipboard...")
            self.driver.execute_script("navigator.clipboard.writeText(arguments[0]);", content)
            self.log("Locating text area and clearing it...")
            textarea_selector = (By.CSS_SELECTOR, 'textarea[aria-label="Text to chunk:"]')
            input_field = wait.until(EC.element_to_be_clickable(textarea_selector))
            input_field.clear()
            self.log("Simulating a 'Paste' (Ctrl+V) command...")
            modifier_key = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL
            input_field.send_keys(modifier_key, "v")
            self.log("Clicking submit button...")
            submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="stBaseButton-secondary"]')))
            submit_button.click()
            json_output = self._extract_json_from_button()
            if json_output:
                return True, json_output, None
            else:
                return False, None, "Failed to extract JSON from the results page."
        except Exception as e:
            return False, None, f"An unexpected error occurred during processing: {e}"
        finally:
            self.cleanup()
            
    def cleanup(self):
        if self.driver:
            self.log("Cleaning up and closing browser instance.")
            self.driver.quit()
            self.log("✅ Browser closed.")

# --- Component 3: AI Processing Functions ---

def extract_big_chunks(json_data):
    """Extract and format big chunks for AI processing."""
    try:
        big_chunks = json_data.get('big_chunks', [])
        chunks = []
        
        for chunk in big_chunks:
            chunk_index = chunk.get('big_chunk_index', len(chunks) + 1)
            small_chunks = chunk.get('small_chunks', [])
            
            # Join small chunks with newlines
            joined_text = '\n'.join(small_chunks)
            
            chunks.append({
                "index": chunk_index,
                "text": joined_text,
                "count": len(small_chunks)
            })
        
        return chunks
    except Exception as e:
        logger.error(f"Error extracting big chunks: {e}")
        return []

async def call_assistant(api_key, assistant_id, content, chunk_index):
    """Call OpenAI Assistant API for chunk analysis."""
    try:
        client = OpenAI(api_key=api_key)
        
        # Create thread
        thread = client.beta.threads.create()
        logger.info(f"Thread created: {thread.id}")
        
        # Add message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content
        )
        
        # Run assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        # Poll for completion
        while run.status in ['queued', 'in_progress']:
            await asyncio.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            # Get response
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            response_content = messages.data[0].content[0].text.value
            
            logger.info(f"Assistant call completed successfully for chunk {chunk_index}")
            return {
                "success": True,
                "content": response_content,
                "chunk_index": chunk_index
            }
        else:
            logger.error(f"Assistant run failed with status: {run.status}")
            return {
                "success": False,
                "error": f"Assistant run failed: {run.status}",
                "chunk_index": chunk_index
            }
            
    except Exception as e:
        logger.error(f"Error calling assistant for chunk {chunk_index}: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunk_index": chunk_index
        }

async def process_chunks_parallel(chunks, api_key):
    """Process all chunks in parallel using OpenAI Assistant."""
    try:
        tasks = []
        for chunk in chunks:
            task = call_assistant(
                api_key=api_key,
                assistant_id=ANALYZER_ASSISTANT_ID,
                content=chunk["text"],
                chunk_index=chunk["index"]
            )
            tasks.append(task)
        
        # Execute all tasks simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
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
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        return [{"success": False, "error": str(e), "chunk_index": 0}]

def create_final_report_simple(analysis_results):
    """Create final report by simple concatenation."""
    try:
        report_parts = []
        
        # Add header
        audit_date = datetime.now().strftime("%Y-%m-%d")
        header = f"""# YMYL Compliance Audit Report

**Audit Date:** {audit_date}
**Content Type:** Online Casino/Gambling  
**Analysis Method:** Section-by-section E-E-A-T compliance review

---

"""
        report_parts.append(header)
        
        # Add successful analyses
        successful_count = 0
        error_count = 0
        
        for result in analysis_results:
            if result.get("success"):
                report_parts.append(result["content"])
                report_parts.append("\n---\n")
                successful_count += 1
            else:
                error_count += 1
                error_section = f"""
# Analysis Error for Chunk {result.get('chunk_index', 'Unknown')}

❌ **Processing Failed**
Error: {result.get('error', 'Unknown error')}

---
"""
                report_parts.append(error_section)
        
        # Add processing summary
        total_sections = successful_count + error_count
        summary = f"""
## Processing Summary
**✅ Sections Successfully Analyzed:** {successful_count}
**❌ Sections with Analysis Errors:** {error_count}  
**📊 Total Sections:** {total_sections}

---
*Report generated by AI-powered YMYL compliance analysis system*
"""
        report_parts.append(summary)
        
        return ''.join(report_parts)
        
    except Exception as e:
        logger.error(f"Error creating final report: {e}")
        return f"Error generating report: {e}"

# --- Main Workflow Function ---
def process_url_workflow_with_logging(url, log_callback=None):
    result = {'success': False, 'url': url, 'extracted_content': None, 'json_output': None, 'error': None}
    
    def log(message):
        if log_callback: log_callback(message)
        logger.info(message)
        
    try:
        log("🚀 Initializing content extractor...")
        extractor = ContentExtractor()
        log(f"🔍 Fetching and extracting content from: {url}")
        success, content, error = extractor.extract_content(url)
        if not success:
            result['error'] = f"Content extraction failed: {error}"; return result
        result['extracted_content'] = content
        log(f"✅ Content extracted: {len(content):,} characters")

        log("🤖 Initializing chunk processor...")
        processor = ChunkProcessor(log_callback=log)
        success, json_output, error = processor.process_content(content)
        if not success:
            result['error'] = f"Chunk processing failed: {error}"; return result
        
        result['json_output'] = json_output
        result['success'] = True
        log("🎉 Workflow Complete!")
        return result
    except Exception as e:
        log(f"💥 An unexpected error occurred in the workflow: {str(e)}")
        result['error'] = f"An unexpected workflow error occurred: {str(e)}"
        return result

async def process_ai_analysis(json_output, api_key, log_callback=None):
    """Process AI compliance analysis on chunked content."""
    def log(message):
        if log_callback: log_callback(message)
        logger.info(message)
    
    try:
        # Parse JSON and extract chunks
        log("📊 Parsing JSON and extracting chunks...")
        json_data = json.loads(json_output)
        chunks = extract_big_chunks(json_data)
        
        if not chunks:
            return False, "No chunks found in JSON data", None
            
        log(f"🚀 Starting parallel analysis of {len(chunks)} chunks...")
        log(f"- Analyzer: {ANALYZER_ASSISTANT_ID}")
        log("- Report Maker: Simple Concatenation (No AI)")
        log("- API Key Status: ✅ Loaded")
        
        # Process chunks in parallel
        analysis_results = await process_chunks_parallel(chunks, api_key)
        
        # Create final report
        log("📝 Assembling final report...")
        final_report = create_final_report_simple(analysis_results)
        
        log("🎉 AI Analysis Complete!")
        return True, final_report, analysis_results
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {e}"
        log(f"❌ {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"AI analysis error: {e}"
        log(f"❌ {error_msg}")
        return False, error_msg, None

# --- Streamlit UI ---
def main():
    st.title("🔄 Content Processor with AI Analysis")
    st.markdown("**Automatically extract content from websites, generate JSON chunks, and perform YMYL compliance analysis**")
    
    # Sidebar configuration
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=True, help="Show detailed processing logs")
    
    # API Key configuration
    st.sidebar.markdown("### 🔑 AI Analysis Configuration")
    api_key = None
    
    # Try to get API key from secrets first
    try:
        api_key = st.secrets.get("openai_api_key")
        if api_key:
            st.sidebar.success("✅ API Key loaded from secrets")
        else:
            raise KeyError("No API key in secrets")
    except (KeyError, AttributeError):
        # Fallback to user input
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
        url = st.text_input(
            "Enter the URL to process:",
            placeholder="https://www.casinohawks.com/bonuses/bonus-code",
            help="Enter a complete URL including http:// or https://"
        )
        if st.button("🚀 Process URL", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a URL to process")
                return
            
            with st.spinner("Processing your request... This may take several minutes for large content."):
                log_placeholder = st.empty()
                log_messages = []

                def log_callback(message):
                    # Fetches current time based on user's location
                    utc_now = datetime.now(pytz.utc)
                    cest_tz = pytz.timezone('Europe/Malta')
                    cest_now = utc_now.astimezone(cest_tz)
                    log_messages.append(f"`{cest_now.strftime('%H:%M:%S')} (CEST)`: {message}")
                    with log_placeholder.container():
                        st.info("\n\n".join(log_messages))
                
                # Clear previous results before starting a new run
                if 'latest_result' in st.session_state:
                    del st.session_state['latest_result']
                if 'ai_analysis_result' in st.session_state:
                    del st.session_state['ai_analysis_result']
                
                result = process_url_workflow_with_logging(url, log_callback if debug_mode else None)
                st.session_state['latest_result'] = result

                if result['success']:
                    st.success("Processing completed successfully!")
                else:
                    st.error(f"An error occurred: {result['error']}")

    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1.  **Extract**: Scrapes content from H1, Subtitle, Lead, Article, FAQ, and Author sections.
        2.  **Copy & Paste**: Submits the text to `chunk.dejan.ai` using a robust copy-paste simulation.
        3.  **Monitor**: Waits for the `<h3>` result heading to appear.
        4.  **Extract JSON**: Securely polls and extracts the complete JSON from the copy button's data attribute.
        5.  **AI Analysis**: Process chunks with OpenAI for YMYL compliance review.
        6.  **Display**: Shows results in the tabs below.
        """)
        st.info("💡 **New**: This version includes AI-powered YMYL compliance analysis!")

    # Results Display
    if 'latest_result' in st.session_state and st.session_state['latest_result'].get('success'):
        result = st.session_state['latest_result']
        st.markdown("---")
        st.subheader("📊 Results")
        
        # AI Analysis Button
        if api_key and st.button("🤖 Process with AI Compliance Analysis", type="secondary", use_container_width=True):
            with st.spinner("Running AI compliance analysis... This may take several minutes."):
                ai_log_placeholder = st.empty()
                ai_log_messages = []

                def ai_log_callback(message):
                    utc_now = datetime.now(pytz.utc)
                    cest_tz = pytz.timezone('Europe/Malta')
                    cest_now = utc_now.astimezone(cest_tz)
                    ai_log_messages.append(f"`{cest_now.strftime('%H:%M:%S')} (CEST)`: {message}")
                    with ai_log_placeholder.container():
                        st.info("\n\n".join(ai_log_messages))

                # Run AI analysis
                success, ai_result, analysis_details = await process_ai_analysis(
                    result['json_output'], 
                    api_key, 
                    ai_log_callback if debug_mode else None
                )
                
                if success:
                    st.session_state['ai_analysis_result'] = {
                        'success': True,
                        'report': ai_result,
                        'details': analysis_details
                    }
                    st.success("✅ AI compliance analysis completed!")
                else:
                    st.session_state['ai_analysis_result'] = {
                        'success': False,
                        'error': ai_result
                    }
                    st.error(f"❌ AI analysis failed: {ai_result}")

        # Tabs for results
        if 'ai_analysis_result' in st.session_state and st.session_state['ai_analysis_result'].get('success'):
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 AI Compliance Report", "📊 Analysis Details", "🔧 JSON Output", "📄 Extracted Content", "📈 Summary"])
            
            with tab1:
                st.markdown("### YMYL Compliance Analysis Report")
                ai_report = st.session_state['ai_analysis_result']['report']
                st.code(ai_report, language='markdown')
                st.download_button(
                    label="💾 Download Compliance Report",
                    data=ai_report,
                    file_name=f"ymyl_compliance_report_{int(time.time())}.md",
                    mime="text/markdown"
                )
                
                with st.expander("📖 View Formatted Report"):
                    st.markdown(ai_report)
            
            with tab2:
                st.markdown("### Individual Analysis Results")
                analysis_details = st.session_state['ai_analysis_result']['details']
                for detail in analysis_details:
                    chunk_idx = detail.get('chunk_index', 'Unknown')
                    if detail.get('success'):
                        with st.expander(f"✅ Chunk {chunk_idx} Analysis (Success)"):
                            st.markdown(detail['content'])
                    else:
                        with st.expander(f"❌ Chunk {chunk_idx} Analysis (Error)"):
                            st.error(f"Error: {detail.get('error', 'Unknown error')}")
            
            with tab3:
                st.code(result['json_output'], language='json')
                st.download_button(
                    label="💾 Download JSON",
                    data=result['json_output'],
                    file_name=f"chunks_{int(time.time())}.json",
                    mime="application/json"
                )
            
            with tab4:
                st.text_area("Raw extracted content:", value=result['extracted_content'], height=400)
            
            with tab5:
                st.subheader("Processing Summary")
                try:
                    json_data = json.loads(result['json_output'])
                    big_chunks = json_data.get('big_chunks', [])
                    total_small_chunks = sum(len(chunk.get('small_chunks', [])) for chunk in big_chunks)
                    
                    colA, colB, colC = st.columns(3)
                    colA.metric("Big Chunks", len(big_chunks))
                    colB.metric("Total Small Chunks", total_small_chunks)
                    colC.metric("Content Length", f"{len(result['extracted_content']):,} chars")
                    
                    # AI Analysis Summary
                    if 'ai_analysis_result' in st.session_state and st.session_state['ai_analysis_result'].get('success'):
                        analysis_details = st.session_state['ai_analysis_result']['details']
                        successful_analyses = sum(1 for detail in analysis_details if detail.get('success'))
                        failed_analyses = len(analysis_details) - successful_analyses
                        
                        st.markdown("#### AI Analysis Summary")
                        colD, colE, colF = st.columns(3)
                        colD.metric("Successful Analyses", successful_analyses, delta=None)
                        colE.metric("Failed Analyses", failed_analyses, delta=None if failed_analyses == 0 else f"-{failed_analyses}")
                        colF.metric("Analysis Coverage", f"{(successful_analyses/len(analysis_details)*100):.1f}%")
                        
                except (json.JSONDecodeError, TypeError):
                    st.warning("Could not parse JSON for statistics.")
                st.info(f"**Source URL**: {result['url']}")
        else:
            # Show original tabs without AI analysis
            tab1, tab2, tab3 = st.tabs(["🎯 JSON Output", "📄 Extracted Content", "📈 Summary"])
            
            with tab1:
                st.code(result['json_output'], language='json')
                st.download_button(
                    label="💾 Download JSON",
                    data=result['json_output'],
                    file_name=f"chunks_{int(time.time())}.json",
                    mime="application/json"
                )
            with tab2:
                st.text_area("Raw extracted content:", value=result['extracted_content'], height=400)
            with tab3:
                st.subheader("Processing Summary")
                try:
                    json_data = json.loads(result['json_output'])
                    big_chunks = json_data.get('big_chunks', [])
                    total_small_chunks = sum(len(chunk.get('small_chunks', [])) for chunk in big_chunks)
                    
                    colA, colB, colC = st.columns(3)
                    colA.metric("Big Chunks", len(big_chunks))
                    colB.metric("Total Small Chunks", total_small_chunks)
                    colC.metric("Content Length", f"{len(result['extracted_content']):,} chars")
                except (json.JSONDecodeError, TypeError):
                    st.warning("Could not parse JSON for statistics.")
                st.info(f"**Source URL**: {result['url']}")
        
        # Show API key reminder if not available
        if not api_key:
            st.info("💡 **Tip**: Add your OpenAI API key to enable AI compliance analysis!")

if __name__ == "__main__":
    main()
