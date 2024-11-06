import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Adithya's Cold Email Generator: AI-Powered Solutions for Personalized Outreach")
    
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-39013")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Load the webpage and extract job data
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            
            # Load the portfolio and extract job postings
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            
            # Display job results and generate email
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                
                # Show generated email
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    # Instantiate Chain and Portfolio classes
    chain = Chain()
    portfolio = Portfolio()
    
    # Configure Streamlit layout and title
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    
    # Start the app
    create_streamlit_app(chain, portfolio, clean_text)
