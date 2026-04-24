import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.graph.nodes import db

def ingest_file(uploaded_file):
    # Save the uploaded file to a temporary location
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Load the document based on its type
        if ext.lower() == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
            
        docs = loader.load()
        
        # Split the document into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Add metadata to chunks so we know where they came from
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
            
        # Add the chunks to our Qdrant vector database
        db.add_documents(chunks)
        
        return len(chunks)
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)
