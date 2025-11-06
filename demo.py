from mlx_lm import load, generate
import json

model, tokenizer = load("Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path="adapters")

def generate_sql(question, database):
    prompt = f"Generate a SQL query to answer the following question.\n\nDatabase: {database}\nQuestion: {question}"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=512, verbose=False)

# Demo
print("DEMO: Fine-tuned Text-to-SQL Model\n")
sql = generate_sql("How many users visited in January?", "ga4")
print(f"Generated SQL:\n{sql}")