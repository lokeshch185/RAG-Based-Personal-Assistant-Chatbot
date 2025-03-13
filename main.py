from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

file_path = "./profile.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(all_splits, embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-1.5",
    convert_system_message_to_human=True,
    temperature=0.5,
    max_tokens=200,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for answering on behalf of a person. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Context: {context}"),
    ("human", """Question: {question}  """)
])

response_chain = prompt | llm | StrOutputParser()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')

    if not query:
        return jsonify({'success': False, 'message': 'No message provided'}), 400

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query, k=1, return_documents=False
        )
        Document, relevance_score = results[0]
        context = Document.page_content

        response = response_chain.invoke({"question": query, "context": context})
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

