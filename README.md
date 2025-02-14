# 📚 AI & RAG-powered Chatbot

## 🔍 What is this?
This chatbot combines **Retrieval-Augmented Generation (RAG)** with AI to answer questions about research papers. It fetches relevant information from a document and generates responses using a **Groq LLaMA-3 model**. 

---

## ⚙️ How to Set It Up

### ✅ What You Need
- **Python 3.8+** installed on your system.
- A virtual environment (**optional but recommended**).
- A **Groq API key** stored in a `.env` file.

### 🚀 Installation Steps
1. **Clone the repository:**  
   ```sh
   git clone https://github.com/hxtanishq/AI-RAG-ChatBot.git
   cd AI-RAG-ChatBot
   ```
2. **Create and activate a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up your environment variables:**  
   - Create a `.env` file in the project folder.
   - Add your Groq API key:
     ```sh
     GROQ_API_KEY=your_api_key_here
     ```
5. **Run the chatbot:**  
   ```sh
   streamlit run app.py
   ```

---

## 🛠️ How It Works

### 🗂️ Step 1: Retrieving Information
1. **Loads the research paper** from the provided URL.
2. **Breaks it into chunks** to make searching efficient.
3. **Cleans and preprocesses the text** (removes stopwords, tokenizes, lemmatizes).
4. **Stores chunk embeddings in FAISS** for quick retrieval.
5. **Uses hybrid search** (semantic + keyword matching) to find the most relevant parts of the document.

### 🤖 Step 2: Generating Answers
1. **Processes the user query** similarly to document chunks.
2. **Finds the best matching chunks** using FAISS and keyword overlap.
3. **Builds a structured prompt** and sends it to the **Groq LLaMA-3-70b model**.
4. **Generates a response** based on the retrieved information.
5. **Avoids hallucination** by returning "Answer not found" when necessary.

---

## 🖼️ Sample Screenshot
![Chatbot Screenshot](AI-RAG-ChatBot\image.png)  
_A quick look at the chatbot in action!_

---

This setup ensures **high-quality, contextually accurate answers** while preventing incorrect information. Happy coding! 🚀

