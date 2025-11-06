# demo.py
from mlx_lm import load, generate

print("Loading model...")
model, tokenizer = load("Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path="adapters")
print("Model loaded!\n")

def generate_sql(question, database):
    prompt = f"Generate a SQL query to answer the following question.\n\nDatabase: {database}\nQuestion: {question}"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=512, verbose=False)

print("="*60)
print("TEXT2SQL DEMO")
print("="*60)

print("\nQuestion:")
question = input("> ")

print("Database (make one up):")
database = input("> ")

print("\nGenerated SQL:")
print("-"*60)
sql = generate_sql(question, database)
print(sql)
print("-"*60)