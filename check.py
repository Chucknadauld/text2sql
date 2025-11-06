import json
import os

# Check if there's an eval set
eval_file = "Spider2/spider2-lite/evaluation_suite/spider2-lite_eval.jsonl"

if os.path.exists(eval_file):
    eval_data = []
    with open(eval_file) as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    print(f"Evaluation set found: {len(eval_data)} examples")
    print(f"First eval example ID: {eval_data[0]['instance_id']}")
    
    # Check if eval has gold SQL
    gold_sql_dir = "Spider2/spider2-lite/evaluation_suite/gold/sql"
    eval_ids = [ex['instance_id'] for ex in eval_data]
    sql_files = set([f.replace('.sql', '') for f in os.listdir(gold_sql_dir) if f.endswith('.sql')])
    
    eval_with_sql = [eid for eid in eval_ids if eid in sql_files]
    print(f"Eval examples with gold SQL: {len(eval_with_sql)}")
else:
    print("No eval file found")

# Create the training dataset
print("\n" + "="*60)
print("CREATING TRAINING DATASET")
print("="*60)

json_file = "Spider2/spider2-lite/spider2-lite.jsonl"
data = []
with open(json_file) as f:
    for line in f:
        data.append(json.loads(line))

gold_sql_dir = "Spider2/spider2-lite/evaluation_suite/gold/sql"
sql_files = set([f.replace('.sql', '') for f in os.listdir(gold_sql_dir) if f.endswith('.sql')])

# Create training pairs
training_data = []
for ex in data:
    instance_id = ex['instance_id']
    if instance_id in sql_files:
        sql_path = os.path.join(gold_sql_dir, f"{instance_id}.sql")
        with open(sql_path, 'r') as f:
            sql_content = f.read().strip()
        
        training_data.append({
            'instance_id': instance_id,
            'question': ex['question'],
            'db': ex['db'],
            'external_knowledge': ex.get('external_knowledge', ''),
            'sql': sql_content
        })

print(f"\nCreated {len(training_data)} training examples")
print(f"\nFirst training example:")
print(f"  ID: {training_data[0]['instance_id']}")
print(f"  Question: {training_data[0]['question'][:100]}...")
print(f"  Database: {training_data[0]['db']}")
print(f"  SQL (first 100 chars): {training_data[0]['sql'][:100]}...")

# Save to JSON
output_file = "training_data.json"
with open(output_file, 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"\nSaved training data to: {output_file}")