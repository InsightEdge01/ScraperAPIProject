import os
import time
import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader,UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun


# Load environment variables
load_dotenv()
SCRAPER_API_KEY = os.getenv('SCRAPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Base URLs for scraping
ZILLOW_URL = "https://www.zillow.com/houston-tx/{}_p/"
HOMES_URL = "https://www.homes.com/houston-tx/{}_p/"
REDFIN_URL = "https://www.redfin.com/city/8903/TX/Houston/{}_p/"
OUTPUT_FILE = "scraped_properties.txt"

def scrape_listings(base_url, site_name, max_pages=15):
    """Scrapes property listings from real estate websites."""
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        for page in range(1, max_pages + 1):
            page_url = base_url.format(page)
            print(f"Scraping {site_name} - Page {page}: {page_url}")

            for attempt in range(3):  # Retry up to 3 times if necessary
                try:
                    payload = {
                        'api_key': SCRAPER_API_KEY,
                        'url': page_url,
                        'country': 'us',
                        'output_format': 'text'
                    }
                    
                    response = requests.get('https://api.scraperapi.com/', params=payload)
                    
                    if response.status_code == 200:
                        file.write(f"\n{site_name} - Page {page}:\n")
                        file.write(response.text + "\n")  # Save scraped data to file
                        break  # Exit retry loop on success
                    else:
                        print(f"Failed attempt {attempt + 1} for {site_name} page {page}: {response.status_code}")
                except Exception as e:
                    print(f"Error scraping {site_name} page {page}: {e}")


def scrape_all_sites():
    """Triggers scraping for all real estate platforms."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("Scraped Listings:\n")
    scrape_listings(ZILLOW_URL, "Zillow")
    scrape_listings(HOMES_URL, "Homes.com")
    scrape_listings(REDFIN_URL, "Redfin.com")

def embed_property_data():
    """Creates FAISS vector store from scraped property data."""
    try:
        loader = UnstructuredFileLoader(OUTPUT_FILE) 
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    except UnicodeDecodeError as e:
        print(f"Error decoding file: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while embedding data: {e}")
        raise

def create_prompt_template():
    """Defines the prompt template for OpenAI."""
    template = """You are an expert real estate assistant. You have a collection of properties from Redfin.com, Zillow.com, and Homes.com.
    Answer property-related questions accurately based on the listings in Houston.

    User Query:
    Location: {location}
    Price Range: {price}
    Bedrooms: {beds}
    Bathrooms: {baths}
    Square Footage: {sqft}

    Provide a list of suitable properties including and All relevant insights from the Real estate data provided from Redfin.com, Zillow.com, and Homes.com.Include the listing Source and Address Where the Property is located"""
    
    return PromptTemplate(input_variables=["location", "price", "beds", "baths", "sqft"], template=template)

def get_recommendations(vector_store, location, price, beds, baths, sqft):
    """Retrieves the top property recommendations based on user query."""
    query = f"Find properties in {location} within {price} range, with {beds} beds, {baths} baths, and {sqft} sqft."
    docs = vector_store.similarity_search(query, k=5)
    recommendations_text = "\n".join([doc.page_content for doc in docs])

    prompt_template = create_prompt_template()
    prompt = prompt_template.format(location=location, price=price, beds=beds, baths=baths, sqft=sqft)

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "You are an expert real estate agent."},
                  {"role": "user", "content": f"Here are some properties: {recommendations_text}. {prompt}"}],
        temperature=0.7
    )

    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="AI-Powered Real Estate Agent", layout="wide")
st.title("🏡 AI-Powered Real Estate Agent")
st.write("Discover top property listings from Zillow.com/Homes.com/Redfin.com based on your preferences!")

# Scrape listings if text file doesn't exist
if not os.path.exists(OUTPUT_FILE):
    with st.spinner("Scraping Zillow, Redfin, and Homes.com listings..."):
        scrape_all_sites()

# Load vector store
if not os.path.exists("vectorstore"):
    with st.spinner("Embedding property data..."):
        vector_store = embed_property_data()
        FAISS.save_local(vector_store, "vectorstore")  # Save vector store
else:
    vector_store = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

st.subheader("🔍 Search for Your Dream Home")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    location = st.text_input("🏙️ Enter Location", "Houston")
with col2:
    price = st.text_input("💲 Enter Price Range", "200000-500000")
with col3:
    beds = st.selectbox("🛏️ Number of Bedrooms", ["Any", "1", "2", "3", "4+"], index=0)
with col4:
    baths = st.selectbox("🛁 Number of Bathrooms", ["Any", "1", "2", "3", "4+"], index=0)
with col5:
    sqft = st.text_input("🏠 Minimum Square Footage", "1000")

# Initialize session state for the question box
if "show_question_box" not in st.session_state:
    st.session_state.show_question_box = False

# Button to find home recommendations
if st.button("🔎 Find Homes"):
    recommendations = get_recommendations(vector_store, location, price, beds, baths, sqft)
    
    # Keep the question section visible
    st.session_state.show_question_box = True  

    if recommendations:
        st.subheader("🏠 Recommended Properties")
        st.write(recommendations)
    else:
        st.warning("No recommendations available at the moment. Please try again later.")

# Function to query the vector store for general real estate questions and pass it through LLM
def answer_general_question(question, vector_store):
    # Retrieve top 3 similar docs from the vector store
    docs = vector_store.similarity_search(question, k=3)
    
    if docs:
        # Concatenate the content of the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prepare the prompt for the LLM
        prompt = f"Provide relevant insights from the Real estate data provided from Redfin.com, Zillow.com, and Homes.com.Answer the following question based on the provided information:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Query the LLM (using gpt-3.5-turbo or gpt-4 here)
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change this to 'gpt-4' if you have access
            messages=[{"role": "system", "content": "You are an assistant who helps answer questions based on real estate listings."},
                      {"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0
        )
        
        # Extract the answer from the LLM's response
        return response.choices[0].message.content
    else:
        return "Sorry, I couldn't find an answer to your question in the available data."

# Additional Question Section
st.subheader("\U0001F914 Have More Questions?")
if "show_question_box" not in st.session_state:
    st.session_state.show_question_box = False

if st.button("Do you have any more questions?"):
    st.session_state.show_question_box = True

if st.session_state.show_question_box:
    question = st.text_input("Ask your question about real estate listings:")
    if st.button("Get Answer"):
        with st.spinner("Searching for the answer..."):
            if question.strip():  # If user entered a question
                response = answer_general_question(question, vector_store)
            else:  # If no question, default to home recommendations
                response = get_recommendations(vector_store, location, price, beds, baths, sqft)
        st.write(response)

#search the web for more real estate information
search = DuckDuckGoSearchRun()
question = st.text_input("Ask any question on the web with real estate")
if st.button("Search the Web"):
            with st.spinner("Searching the web..."):
                if question.strip():
                    web_result = search.invoke(question)
                    st.write("Web Search Result:", web_result)
                else:
                    st.write("Please enter a question before searching the web.")