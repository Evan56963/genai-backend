from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer