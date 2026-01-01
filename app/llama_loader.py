from transformers import LlamaForCausalLM, LlamaTokenizer

def load_llama_model(model_path: str):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    return {"model": model, "tokenizer": tokenizer}