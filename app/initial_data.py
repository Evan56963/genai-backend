import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from sqlmodel import SQLModel, create_engine, Session

def load_llama_model(llama_model_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(

        model_name=llama_model_path, 
        dtype=torch.bfloat16, 
        max_seq_length=1024, 
        load_in_4bit=True,
)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def initialize_chromadb_client(embed_model_name: str, persist_directory: str, embed_model_collection: str):
    embed_model = SentenceTransformer(embed_model_name, device="cuda")
    client = PersistentClient(path=persist_directory)
    collection = client.get_collection(name=embed_model_collection)
    return embed_model, collection

def initialize_db(sqlite_filepath: str) -> Session:
    sqlite_filepath = sqlite_filepath
    sqlite_url = f"sqlite:///{sqlite_filepath}"
    engine = create_engine(sqlite_url, echo=False)
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    return session