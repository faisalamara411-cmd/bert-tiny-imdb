import os
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
run_name = os.getenv("WANDB_RUN_NAME", "bert-imdb-run")
testing = os.getenv("TESTING", "false").lower()
dataset_name = os.getenv("DATASET_NAME", "imdb")

if api_key is None:
    raise ValueError("Please set the WANDB_API_KEY in the .env file")

#  Initialize W&B
wandb.login(key=api_key)
run = wandb.init(project="imdb-bert-tiny", name=run_name, job_type="train")

#  Load Dataset
datasets = load_dataset(dataset_name)
datasets = DatasetDict({
    'train': datasets['train'],
    'test': datasets['test']
})

print(f"Train: {datasets['train'].num_rows}, Test: {datasets['test'].num_rows}")

# Optional: Use a smaller subset for testing/debugging
if testing == "true":
    datasets['train'] = datasets['train'].shuffle(seed=42).select(range(5000))
    datasets['test'] = datasets['test'].shuffle(seed=42).select(range(1000))
    print("Running in TEST mode — using 5,000 training and 1,000 test samples.")


#  Tokenization
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

encoded_datasets = datasets.map(tokenize_function, batched=True)
encoded_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

#  Model
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

#  Evaluation Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

#  Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="wandb",
    run_name=run_name,
    save_total_limit=4
)



# Trainer
from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
trainer.train()

# Save Model & Tokenizer
model.save_pretrained("bert-tiny-imdb")
tokenizer.save_pretrained("bert-tiny-imdb")

# Log model to W&B
model_artifact = wandb.Artifact("bert-tiny-imdb-model", type="model")
model_artifact.add_dir("bert-tiny-imdb")
run.log_artifact(model_artifact)

train_split = datasets['train'].to_csv("train.csv")
test_split = datasets['test'].to_csv("test.csv")

dataset_artifact = wandb.Artifact("dataset", type="dataset")
dataset_artifact.add_file("train.csv")
dataset_artifact.add_file("test.csv")
run.log_artifact(dataset_artifact)

run.link_artifact(artifact=model_artifact,
    target_path="wandb-registry-model/bert-tiny-imdb")

run.finish()
