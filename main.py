from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableMap, RunnablePassthrough, RunnableParallel
from typing import Optional, List ,Dict, Any
from pydantic import BaseModel, Field

load_dotenv()

file_path = "./profile.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(docs)



# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(all_splits, embeddings)
# ids = vector_store.add_documents(documents=all_splits)
# print(ids)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
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

query = "what is leetcode"
results = vector_store.similarity_search_with_relevance_scores(
    query, k=1, return_documents=False
)
print (results)
Document, relevance_score = results[0]
context = Document.page_content
print(context)
response = response_chain.invoke({"question": query, "context": context})
print(response)

