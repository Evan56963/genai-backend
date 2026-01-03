from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import torch

def load_llama_model(llama_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    model = AutoModelForCausalLM.from_pretrained(llama_model_path, dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer

def initialize_chromadb_client(embed_model_name: str, persist_directory: str, embed_model_collection: str):
    embed_model = SentenceTransformer(embed_model_name, device="cuda")
    client = PersistentClient(path=persist_directory)
    collection = client.get_collection(name=embed_model_collection)
    return embed_model, collection