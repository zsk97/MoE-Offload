import torch
from datasets import load_dataset
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, Trainer, TrainingArguments

# 加载模型和分词器
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
model_name = "google/switch-base-8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name).to(device)

# 加载数据集
dataset = load_dataset('cnn_dailymail', '3.0.0')

count = 5
# 预处理函数
def preprocess_function(examples):
    # 对输入文章进行编码
    inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    # 对标签（高亮/摘要）进行编码
    labels = tokenizer(examples['highlights'], max_length=128, truncation=True, padding='max_length', return_tensors='pt')

    inputs = {k: v.squeeze() for k, v in inputs.items()}  # 去掉不必要的维度
    labels = labels['input_ids'].squeeze()
    global count
    if count > 0:
        print(inputs['input_ids'].shape, inputs['attention_mask'].shape, labels.shape)
        count -= 1

    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

# 数据预处理
train_dataset = dataset['train'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    tf32=True,
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=1,
    logging_steps=20,
    num_train_epochs=50,
    load_best_model_at_end=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 开始训练
trainer.train()