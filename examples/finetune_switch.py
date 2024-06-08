import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, Trainer, TrainingArguments

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wmt', help='Dataset to use for fine-tuning')
parser.add_argument('--model_name', type=str, default='google/switch-base-8', help='Model to use for fine-tuning')
parser.add_argument('--ipdb', action='store_true', help='enable debug mode')
args = parser.parse_args()
if args.ipdb:
    from ipdb import set_trace
    set_trace()

# 加载模型和分词器
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name).to(device)

# 根据命令行参数选择数据集
if args.dataset == 'wmt':
    dataset = load_dataset("wmt16", "de-en", split={'train': 'train[:100]', 'test': 'test[:100]'})
    task_type = 'translation'
    test_name = 'test'
elif args.dataset == 'neulab/conala':
    dataset = load_dataset("neulab/conala", split={'train': 'train[:100]', 'test': 'test[:100]'})
    task_type = 'code'
elif args.dataset == 'bigbench/goal_step_wikihow':
    dataset = load_dataset("tasksource/bigbench", "goal_step_wikihow", split={'train': 'train[:100]', 'validation': 'validation[:100]'})
    dataset['test'] = dataset['validation']
    task_type = 'qa'

# 预处理函数，需根据任务类型调整
def preprocess_function(examples):
    if task_type == 'translation':
        inputs = tokenizer([x['de'] for x in examples['translation']], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        labels = tokenizer([x['en'] for x in examples['translation']], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    elif task_type == 'code':
        inputs = tokenizer(examples['intent'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        labels = tokenizer(examples['snippet'], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    elif task_type == 'qa':
        inputs = tokenizer(examples['inputs'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        labels = tokenizer([x[0] for x in examples['targets']], max_length=128, truncation=True, padding='max_length', return_tensors='pt')

    inputs = {k: v.squeeze() for k, v in inputs.items()}
    labels = labels['input_ids'].squeeze()
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

# 数据预处理
train_dataset = dataset['train'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results/" + args.dataset + '_' + args.model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    tf32=True,
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=1,
    logging_steps=20,
    num_train_epochs=1,
    load_best_model_at_end=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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