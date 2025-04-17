import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

def create_streamlit_app(llm, portfolio):
    """Streamlit app for generating cold emails based on job descriptions."""
    st.title("Adithya's Cold Email Generator")

    url_input = st.text_input("Enter a Job Posting URL:", value="https://jobs.nike.com/job/R-39013")
    submit_button = st.button("Generate Email")

    if submit_button:
        try:
            loader = WebBaseLoader(url_input)
            page_data = loader.load()

            if not page_data:
                st.error("No data found at the provided URL.")
                return

            data = clean_text(page_data[0].page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            if not jobs:
                st.warning("No job postings found in the extracted data.")
                return

            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)

                st.subheader(f"Cold Email for {job.get('role', 'Unknown Role')}")
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()

    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio)
