import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core.settings import Settings

DOCS_DIR = 'docs'
INDEX_DIR = 'environment_index'

# Find all PDFs in docs/
pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
if len(pdfs) != 1:
    raise ValueError(f"There must be exactly one PDF in the '{DOCS_DIR}' folder. Found: {pdfs}")

pdf_path = os.path.join(DOCS_DIR, pdfs[0])
print(f"Loading PDF from {pdf_path}...")
documents = PDFReader().load_data(pdf_path)

# Set local embedding model
Settings.embed_model = "local:sentence-transformers/all-MiniLM-L6-v2"

# Build index
print("Building index...")
index = VectorStoreIndex.from_documents(documents)

# Save index
print(f"Saving index to {INDEX_DIR}...")
index.storage_context.persist(persist_dir=INDEX_DIR)
print("Indexing complete.") 