from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


file_path = "./profile.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

