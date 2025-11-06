import json

# Load training data
with open('training_data.json', 'r') as f:
    data = json.load(f)

# Format for Qwen chat template
formatted_data = []
for item in data:
    # Create the prompt
    prompt = f"Generate a SQL query to answer the following question.\n\nDatabase: {item['db']}\nQuestion: {item['question']}"
    
    # Add external knowledge if available
    if item['external_knowledge']:
        prompt += f"\n\nExternal Knowledge: {item['external_knowledge']}"
    
    # Format as chat messages
    formatted_item = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item['sql']}
        ]
    }
    formatted_data.append(formatted_item)

# Split into train/valid (90/10 split)
split_idx = int(len(formatted_data) * 0.9)
train_data = formatted_data[:split_idx]
valid_data = formatted_data[split_idx:]

# Save
with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('valid.jsonl', 'w') as f:
    for item in valid_data:
        f.write(json.dumps(item) + '\n')

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(valid_data)}")
print(f"\nFiles created: train.jsonl, valid.jsonl")