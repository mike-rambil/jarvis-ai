from llama_index.core import StorageContext, load_index_from_storage

INDEX_DIR = 'environment_index'

def load_llamaindex():
    print(f"Loading index from {INDEX_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    return index

def query_llamaindex(index, question):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response) 