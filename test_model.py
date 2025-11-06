from mlx_lm import load, generate

# Load the fine-tuned model
model, tokenizer = load("Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path="adapters")

# Test question
test_question = "How many users visited the website in January 2021?"
test_db = "ga4"

# Format the prompt (same format as training)
prompt = f"Generate a SQL query to answer the following question.\n\nDatabase: {test_db}\nQuestion: {test_question}"

# Format as chat
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate SQL
print("Question:", test_question)
print("\nGenerating SQL...\n")

response = generate(model, tokenizer, prompt=prompt_text, max_tokens=512, verbose=False)
print(response)