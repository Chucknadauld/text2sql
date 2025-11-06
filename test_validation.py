import json
from mlx_lm import load, generate

# Load model
model, tokenizer = load("Qwen/Qwen2.5-Coder-0.5B-Instruct", adapter_path="adapters")

# Load validation data
with open('valid.jsonl', 'r') as f:
    valid_data = [json.loads(line) for line in f]

# Test on first 3 validation examples
for i, item in enumerate(valid_data[:3]):
    user_msg = item['messages'][0]['content']
    gold_sql = item['messages'][1]['content']
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*60}")
    print(f"Prompt:\n{user_msg[:200]}...")
    
    # Generate
    messages = [{"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated_sql = generate(model, tokenizer, prompt=prompt_text, max_tokens=512, verbose=False)
    
    print(f"\nGenerated SQL:\n{generated_sql[:300]}...")
    print(f"\nGold SQL:\n{gold_sql[:300]}...")