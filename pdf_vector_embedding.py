from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# This file is used for embedding a pdf file, running similarity search and returning the result

def rag_context(model:str="BAAI/bge-m3", file_path:str=None, rag_query:str=None):

    # validation
    if file_path is None:
        raise ValueError("Please include a valid file path")
        exit()
    if rag_query is None:
        raise ValueError("Please input a valid query for similiarty search")
        exit()
        
    # Load and split
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        strip_whitespace=True
        )
    chunks = splitter.split_documents(docs)

    # Embed and store
    embeddings = HuggingFaceEmbeddings(
        model=model,
        model_kwargs = {'device': 'mps'} #set mps for macbook M chip
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    results = vectorstore.similarity_search(rag_query, k=3)
    context = "\n".join([r.page_content for r in results])

    # Feed to LLM
    rag_context = f"Context:\n{context}"

    return rag_context

if __name__ == "__main__":
    file_path = "./input/m_an11539_en-us 2025-04-15.pdf"
    rag_query = "Extract all acquirer fees."
    print(rag_context(file_path = file_path, rag_query = rag_query))
