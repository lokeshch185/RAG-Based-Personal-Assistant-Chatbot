from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


file_path = "./profile.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(all_splits, embeddings)
ids = vector_store.add_documents(documents=all_splits)
results = vector_store.similarity_search(
    "query"
)

print(results[0])
