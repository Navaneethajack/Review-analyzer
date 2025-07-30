import streamlit as st
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime

# Set page config
st.set_page_config(page_title="📊 Reviews Analyzer", page_icon="📊", layout="centered")

# Folder to store uploaded files
UPLOAD_FOLDER = "uploaded_docs"
CLEANED_FOLDER = "cleaned_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)

# Cache the Flan-T5 pipeline
@st.cache_resource
def load_flan_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Data cleaning function
def clean_shopify_data(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Data cleaning steps
    # 1. Standardize timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # 2. Fix inconsistent category/product names
    df['Product Category'] = df['Product Category'].str.strip().str.title()
    df['Product Name'] = df['Product Name'].str.strip().str.title()
    
    # 3. Filter out blank/invalid reviews
    df = df.dropna(subset=['Review Content', 'Rating'])
    df = df[df['Review Content'].str.strip() != '']
    
    # 4. Normalize missing fields
    # Impute missing ratings with median
    if df['Rating'].isna().any():
        median_rating = df['Rating'].median()
        df['Rating'] = df['Rating'].fillna(median_rating)
    
    # Flag rows with missing important fields
    df['Is_Valid'] = ~df[['Customer Email', 'Product Name', 'Review Content']].isna().any(axis=1)
    
    # Save cleaned data
    cleaned_path = os.path.join(CLEANED_FOLDER, "cleaned_" + os.path.basename(file_path))
    df.to_csv(cleaned_path, index=False)
    
    return cleaned_path

# Load and split documents into chunks
def load_and_split(file_path):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Create vector index
def create_faiss_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Semantic search helper
def search_docs(index, query):
    results = index.similarity_search(query, k=3)
    return "\n\n".join([r.page_content for r in results]) if results else ""

# UI Title
st.title("📊 Shopify Reviews Analyzer")

# Session state initialization
for key in ["file_uploaded", "file_path", "cleaned_path", "faiss_index", "flan"]:
    if key not in st.session_state:
        st.session_state[key] = None

# File upload UI
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("Upload Shopify Reviews CSV", type=["csv"])
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.file_uploaded = True
        st.session_state.file_path = file_path
        st.success("✅ Document uploaded. Cleaning data...")
        
        # Clean the data
        try:
            cleaned_path = clean_shopify_data(file_path)
            st.session_state.cleaned_path = cleaned_path
            st.success("🧹 Data cleaned successfully!")
            
            # Process the cleaned file
            docs = load_and_split(cleaned_path)
            st.session_state.faiss_index = create_faiss_index(docs)
            st.session_state.flan = load_flan_pipeline()
            st.success("🎯 Ready! Ask questions about the reviews data.")
            
            # Show preview of cleaned data
            df = pd.read_csv(cleaned_path)
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            st.session_state.file_uploaded = False

# Clear button
if st.session_state.file_uploaded:
    if st.button("🧹 Clear Document"):
        try:
            if st.session_state.file_path and os.path.exists(st.session_state.file_path):
                os.remove(st.session_state.file_path)
            if st.session_state.cleaned_path and os.path.exists(st.session_state.cleaned_path):
                os.remove(st.session_state.cleaned_path)
        except Exception as e:
            st.warning(f"⚠️ Failed to delete files: {e}")
        for key in ["file_uploaded", "file_path", "cleaned_path", "faiss_index", "flan"]:
            st.session_state[key] = None
        st.experimental_rerun()

# Question input + response
if st.session_state.file_uploaded:
    st.subheader("Ask Questions About the Reviews")
    user_input = st.text_input("Your question:")
    
    if user_input:
        with st.spinner("Analyzing reviews..."):
            try:
                context = search_docs(st.session_state.faiss_index, user_input)

                # If no context found, use a broader search
                if not context:
                    context = search_docs(st.session_state.faiss_index, "reviews summary")

                prompt = f"""
                You are analyzing Shopify product reviews. Use the following context to answer the question.
                Context:
                {context}
                
                Question: {user_input}
                Answer:"""
                
                response = st.session_state.flan(prompt, max_new_tokens=256)[0]["generated_text"]

                st.markdown("### 📝 Analysis")
                st.write(response)
                
                # Add some common analysis options
                if st.button("Show Overall Sentiment Analysis"):
                    sentiment_prompt = """
                    Based on all reviews, summarize the overall sentiment towards the products.
                    Consider the distribution of ratings and common themes in reviews.
                    Provide specific examples of positive and negative feedback."""
                    
                    sentiment_response = st.session_state.flan(sentiment_prompt, max_new_tokens=512)[0]["generated_text"]
                    st.write(sentiment_response)
                    
                if st.button("Show Product Performance Summary"):
                    product_prompt = """
                    Analyze which products are performing well and which are not based on the reviews.
                    Mention specific products with their average ratings and common feedback points."""
                    
                    product_response = st.session_state.flan(product_prompt, max_new_tokens=512)[0]["generated_text"]
                    st.write(product_response)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add some sample questions
if st.session_state.file_uploaded:
    st.subheader("Try These Sample Questions")
    cols = st.columns(2)
    with cols[0]:
        if st.button("What are common complaints?"):
            st.session_state.user_input = "What are the most common complaints in the reviews?"
    with cols[1]:
        if st.button("Which products have highest ratings?"):
            st.session_state.user_input = "Which products have the highest average ratings?"
    
    cols = st.columns(2)
    with cols[0]:
        if st.button("Trends in customer feedback?"):
            st.session_state.user_input = "What trends can you identify in the customer feedback over time?"
    with cols[1]:
        if st.button("Frequent positive comments?"):
            st.session_state.user_input = "What are the most frequent positive comments in the reviews?"