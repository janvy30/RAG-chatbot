import os
import chromadb

PERSIST_DIR = os.getenv("CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")

# Use PersistentClient so vectors persist between runs
_client = chromadb.PersistentClient(path=PERSIST_DIR)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

def get_collection():
    return _collection
