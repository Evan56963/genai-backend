from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return {"model": model, "tokenizer": tokenizer}