from dotenv import load_dotenv
import os
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb

def get_chroma_collection(chroma_client):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CHROMA_OPENAI_MODEL = os.getenv('CHROMA_OPENAI_MODEL')
    open_ai_embedding = embedding_functions.OpenAIEmbeddingFunction(
      api_key=OPENAI_API_KEY,
      model_name=CHROMA_OPENAI_MODEL
    )
    CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=open_ai_embedding)
    print('collection peek: ',collection.peek()) # returns a list of the first 10 items in the collection
    print('collection count: ',collection.count())

def reset_chroma(reset=False):
    CHROMA_HOST = os.getenv('CHROMA_HOST')
    CHROMA_PORT = os.getenv('CHROMA_PORT')
    ALLOW_RESET  = os.getenv('ALLOW_RESET')
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST, port=int(CHROMA_PORT), settings=Settings(allow_reset=bool(ALLOW_RESET), anonymized_telemetry=False))
    
    if reset:
        chroma_client.reset()

    return chroma_client

def main():
    load_dotenv()
    chroma_client = reset_chroma(True)
    get_chroma_collection(chroma_client)

if __name__=="__main__": 
    main() 
    