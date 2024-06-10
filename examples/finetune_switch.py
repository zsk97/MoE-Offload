# refer to: https://wandb.ai/byyoung3/ml-news/reports/A-Guide-to-DeepSpeed-Zero-With-the-HuggingFace-Trainer--Vmlldzo2ODkwMDc4
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wmt', help='Dataset to use for fine-tuning')
parser.add_argument('--model_name', type=str, default='google/switch-base-128', help='Model to use for fine-tuning')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
parser.add_argument('--ipdb', action='store_true', help='enable debug mode')

def main(args):
    # 设置训练参数
    lr = args.lr
    bs = args.bs
    wd = args.wd

    # 根据命令行参数选择数据集
    if args.dataset == 'wmt':
        dataset = load_dataset("wmt16", "de-en", split={'train': f'train[:1%]', 'test': f'test[:200]'})
        task_type = 'translation'
        test_name = 'test'
    elif args.dataset == 'neulab/conala':
        dataset = load_dataset("neulab/conala", split={'train': 'train[:100]', 'test': 'test[:100]'})
        task_type = 'code'
    elif args.dataset == 'bigbench/goal_step_wikihow':
        dataset = load_dataset("tasksource/bigbench", "goal_step_wikihow", split={'train': 'train[:100]', 'validation': 'validation[:100]'})
        dataset['test'] = dataset['validation']
        task_type = 'qa'

    # 加载模型和分词器
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 预处理函数，需根据任务类型调整
    def preprocess_function(examples):
        if task_type == 'translation':
            inputs = tokenizer([x['de'] for x in examples['translation']], max_length=256, truncation=True, padding='longest', return_tensors='pt')
            labels = tokenizer([x['en'] for x in examples['translation']], max_length=128, truncation=True, padding='longest', return_tensors='pt')
        elif task_type == 'code':
            inputs = tokenizer(examples['intent'], max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            labels = tokenizer(examples['snippet'], max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        elif task_type == 'qa':
            inputs = tokenizer(examples['inputs'], max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            labels = tokenizer([x[0] for x in examples['targets']], max_length=128, truncation=True, padding='max_length', return_tensors='pt')

        inputs = {k: v.squeeze() for k, v in inputs.items()}
        labels = labels['input_ids'].squeeze()
        print(inputs['input_ids'].shape, labels.shape)
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels}

    # 数据预处理
    train_dataset = dataset['train'].map(preprocess_function, batched=True)
    test_dataset = dataset['test'].map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt", padding="longest")

    # 设置训练参数
    model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.decoder_start_token_id = 0
    run_name = args.dataset + '_' + args.model_name.replace('/', '_') + '_lr' + str(lr) + '_bs' + str(bs) + '_wd' + str(wd)
    output_dir="./results/" + args.dataset + '_' + args.model_name.replace('/', '_')
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy="steps",
        learning_rate=lr,
        weight_decay=wd,
        bf16=True,
        tf32=True,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=200,
        num_train_epochs=3,
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        deepspeed="./ds_finetune_switch.json",
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir+'/model_state_dict', state_dict=model.state_dict(), safe_serialization=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    if args.ipdb:
        from ipdb import set_trace
        set_trace()


# transformers==4.41.3
# WANDB_MODE=offline deepspeed --include localhost:2,7 finetune_switch.py --model_name google/switch-base-64
