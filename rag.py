import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import requests, os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TF_ENABLE_ONEDNN_OPTS=0 

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"

class EnhancedTextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text.lower())
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return " ".join(tokens)

class EnhancedRAGSystem:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.chat_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile"  
        )
        self.text_processor = EnhancedTextProcessor()
        self.vector_store = None

    def chunk_text(self, documents, chunk_size=1000, chunk_overlap=400):   
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, text_chunks):
        processed_chunks = []
        for chunk in text_chunks:
            chunk.page_content = self.text_processor.preprocess_text(chunk.page_content)
            processed_chunks.append(chunk)
        
        self.vector_store = FAISS.from_documents(processed_chunks, self.embedding_model)
        return self.vector_store

    def hybrid_search(self, query, k=10):
        processed_query = self.text_processor.preprocess_text(query)
        semantic_results = self.vector_store.similarity_search(processed_query, k=k)
    
        keyword_scores = []
        query_terms = set(processed_query.split())
        
        for doc in semantic_results:
            doc_terms = set(doc.page_content.split())
            overlap = len(query_terms.intersection(doc_terms))
            keyword_scores.append(overlap / len(query_terms))
         
        combined_results = []
        for doc, keyword_score in zip(semantic_results, keyword_scores):
            combined_results.append((doc, keyword_score))
         
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in combined_results]

def main():
    st.title("📚 AI & RAG-powered Chatbot")
    
    rag_system = EnhancedRAGSystem()
 
    @st.cache_resource
    def initialize_system():
        temp_pdf_path = "research_paper.pdf"
        response = requests.get(PDF_URL)
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        loader = PyMuPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_chunks = rag_system.chunk_text(documents)
        vector_store = rag_system.create_vector_store(text_chunks)
        return vector_store

    vector_store = initialize_system()
    rag_system.vector_store = vector_store

    query = st.text_area("Enter your query", "Give simple introduction of research paper")
    
    if query:
        retrieved_docs = rag_system.hybrid_search(query)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""Based on the following context, provide a detailed and accurate answer to the question.
        Context: {retrieved_text}
        Question: {query}
        Please ensure your response is:
        1. Directly related to the context
        2. Comprehensive yet concise
        3. Well-structured
        4.Do not give response which are not in context, simply reply answer not found
        Answer:
        """

        messages = [HumanMessage(content=prompt)]
        response = rag_system.chat_model(messages)

        st.subheader("🤖 AI Response")
        st.write(response.content)

        # Calculate and display enhanced metrics
        st.subheader("📊 Retrieval Metrics")
        reference_tokens = set(rag_system.text_processor.preprocess_text(query).split())
        retrieved_tokens = set(retrieved_text.split())
        recall = len(reference_tokens.intersection(retrieved_tokens)) / len(reference_tokens)
        st.write(f"Recall Score: {recall:.2f}")

if __name__ == "__main__":
    main()