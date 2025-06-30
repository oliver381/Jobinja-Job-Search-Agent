import time
from typing import TypedDict
from jobinja_title_generator import JobTitleGenerator
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph

# Initialize Streamlit app
st.set_page_config(
    page_title="Jobinja Job Search Agent",
    page_icon="💼",
    layout="wide"
)
st.title("دستیار هوشمند جستجوی شغل در Jobinja")

# Custom CSS for RTL and styling
st.markdown("""
    <style>
        body {
            direction: rtl;
            text-align: right;
        }
        .stTextInput input {
            text-align: right;
        }
        .job-result {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .job-title {
            font-weight: bold;
            color: #1e88e5;
        }
        .job-company {
            color: #4a5568;
        }
        .job-location {
            color: #718096;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize the Qwen model for generating search keyword
ollama_llm = OllamaLLM(model='qwen2.5:latest', top_p=0.9, temperature=0.7, repeat_penalty=1.1)

# Initialize Selenium WebDriver with persistent browser
def init_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Set path to your chromedriver
    service = Service()
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Define the agent state
class AgentState(TypedDict):
    job_tags: str
    job_skills: str
    generated_title: str
    search_keyword: str
    search_results: list
    driver: webdriver.Chrome

# Define agent nodes
def generate_job_title(state: AgentState):
    """Generate job title using the fine-tuned model"""
    try:
        with st.spinner("در حال تولید عنوان شغل..."):
            job_tags = state['job_tags']
            job_skills = state['job_skills']
            
            title_model = JobTitleGenerator()
            generated_title = title_model.generate_title(job_tags, job_skills)
            
            if not generated_title:
                raise ValueError("مدل نتوانست عنوان شغل تولید کند")
                
            st.session_state['generated_title'] = generated_title
            return {"generated_title": generated_title}
    except Exception as e:
        st.error(f"خطا در تولید عنوان شغل: {str(e)}")
        return {"generated_title": "خطا در تولید عنوان"}

def generate_search_keyword(state: AgentState):
    """Generate search keyword using Ollama LLM"""
    try:
        with st.spinner("در حال تولید کلمه کلیدی..."):
            prompt = (
                f"بر اساس این اطلاعات یک کلمه کلیدی مناسب برای جستجوی شغل در سایت jobinja.ir تولید کن:\n"
                f"عنوان شغل: {state['generated_title']}\n"
                f"برچسب‌ها: {state['job_tags']}\n"
                f"مهارت‌ها: {state['job_skills']}\n\n"
                "کلمه کلیدی باید:\n"
                "- حداکثر دو کلمه باشد\n"
                "- به زبان فارسی باشد\n"
                "- خاص و دقیق باشد\n"
                "- مناسب بازار کار ایران باشد\n\n"
                "فقط خود کلمه کلیدی را بدون توضیح برگردان."
            )
            
            response = ollama_llm.invoke(prompt)
            
            # Clean and extract the first 2 words
            keyword = ' '.join(response.strip().split()[:2])
            keyword = keyword.replace('"', '').replace("'", "").strip()
            
            st.session_state['search_keyword'] = keyword
            return {"search_keyword": keyword}
    except Exception as e:
        st.error(f"خطا در تولید کلمه کلیدی: {str(e)}")
        return {"search_keyword": "خطا در تولید کلمه کلیدی"}

def perform_job_search(state: AgentState):
    """Perform job search on jobinja.ir using Selenium"""
    try:
        with st.spinner("در حال جستجو در Jobinja..."):
            if 'driver' not in state or state['driver'] is None:
                state['driver'] = init_webdriver()
            
            driver = state['driver']
            driver.get("https://jobinja.ir")
            
            # Accept cookies if the popup appears
            try:
                cookie_accept = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "cookie-law-btn"))
                )
                cookie_accept.click()
            except:
                pass
            
            # Wait for search input and enter keyword
            search_input = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CLASS_NAME, "c-jobSearchTop__blockInput"))
            )
            search_input.clear()
            for char in state['search_keyword']:
                search_input.send_keys(char)
                time.sleep(0.1)
            
            # Click the correct search button
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.c-btn.c-btn--secondary2.c-jobSearchTop__submitButton"))
            )
            search_button.click()
            
            # Wait for results container to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".o-listView__item"))
            )
            
            # Scroll to load more results
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            
            # Get job listings container
            job_listings = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".o-listView__item"))
            )
            results = []
            
            for job in job_listings[:10]:  # Get top 10 results
                try:
                    # Extract title and link
                    title_element = WebDriverWait(job, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".c-jobListView__title a"))
                    )
                    title = title_element.text
                    link = title_element.get_attribute("href")
                    
                    # Extract company and location
                    meta_items = WebDriverWait(job, 5).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".c-jobListView__metaItem"))
                    )
                    company = meta_items[0].text if len(meta_items) > 0 else "نامشخص"
                    location = meta_items[1].text if len(meta_items) > 1 else "نامشخص"
                    
                    results.append({
                        "title": title,
                        "company": company,
                        "location": location,
                        "link": link
                    })
                except Exception as e:
                    print(f"Error parsing job listing: {str(e)}")
                    continue
            
            st.session_state['search_results'] = results
            return {"search_results": results}
    except Exception as e:
        st.error(f"خطا در جستجوی شغل: {str(e)}")
        return {"search_results": []}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_job_title", generate_job_title)
workflow.add_node("generate_search_keyword", generate_search_keyword)
workflow.add_node("perform_job_search", perform_job_search)

# Define edges
workflow.add_edge("generate_job_title", "generate_search_keyword")
workflow.add_edge("generate_search_keyword", "perform_job_search")
workflow.add_edge("perform_job_search", END)

# Set entry point
workflow.set_entry_point("generate_job_title")

# Compile the graph
agent = workflow.compile()

# Streamlit UI
def main():
    with st.form("job_search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_tags = st.text_area(
                "برچسب‌های شغلی (با کاما جدا کنید)",
                placeholder="مثال: برنامه‌نویسی, توسعه وب, پایتون",
                height=100
            )
        
        with col2:
            job_skills = st.text_area(
                "مهارت‌های مورد نیاز (با کاما جدا کنید)",
                placeholder="مثال: Django, Flask, HTML, CSS",
                height=100
            )
        
        submitted = st.form_submit_button("جستجوی هوشمند")
    
    if submitted:
        if not job_tags or not job_skills:
            st.warning("لطفاً برچسب‌ها و مهارت‌های شغلی را وارد کنید")
        else:
            # Initialize driver for this session
            if 'driver' not in st.session_state:
                st.session_state.driver = init_webdriver()
            
            # Run the agent
            result = agent.invoke({
                "job_tags": job_tags,
                "job_skills": job_skills,
                "driver": st.session_state.driver
            })
            
            # Display results in an expandable section
            with st.expander("نتایج جستجو", expanded=True):
                st.markdown(f"**عنوان تولید شده:** `{result['generated_title']}`")
                st.markdown(f"**کلمه کلیدی جستجو:** `{result['search_keyword']}`")
                
                st.markdown("---")
                st.subheader("موقعیت‌های شغلی یافت شده")
                
                if not result['search_results']:
                    st.warning("نتیجه‌ای یافت نشد. لطفاً کلمات کلیدی را تغییر دهید.")
                else:
                    for i, job in enumerate(result['search_results'], 1):
                        with st.container():
                            st.markdown(f"""
                                <div class="job-result">
                                    <div class="job-title">{i}. {job['title']}</div>
                                    <div class="job-company">شرکت: {job['company']}</div>
                                    <div class="job-location">موقعیت مکانی: {job['location']}</div>
                                    <div><a href="{job['link']}" target="_blank">مشاهده جزئیات</a></div>
                                </div>
                            """, unsafe_allow_html=True)
    
    # Clean up when done
    if 'driver' in st.session_state and st.session_state.driver:
        st.session_state.driver.quit()
        del st.session_state.driver

if __name__ == "__main__":
    main()