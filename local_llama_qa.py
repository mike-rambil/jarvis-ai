from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys

def load_llama_1b_instruct():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

def ask_llama(question, context=None, generator=None, max_new_tokens=128):
    if context:
        prompt = f"<|user|>\n{context}\n{question}\n<|assistant|>\n"
    else:
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True)
    return result[0]['generated_text']

if __name__ == "__main__":
    generator = load_llama_1b_instruct()
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = input("Enter your question: ")
    answer = ask_llama(question, generator=generator)
    print("\nModel answer:\n", answer) 