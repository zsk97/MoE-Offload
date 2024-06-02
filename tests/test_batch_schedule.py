import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import MoEModelOutput
from MoEOffload.scheduler import scheduler, key_value_select_batch, key_value_select_merge, key_value_order_merge
from MoEOffload.load_utils import process_schedule_dataset, truncate_input

if __name__ == "__main__":

    torch.manual_seed(1234)

    device = "cuda:0"

    in_order = True
    cache_size = 8
    batch_size = 64
    schedule_size = 128
    total_batch_id = 1
    max_new_tokens = 16
    num_experts_per_layer = 64
    top_n = 1
    
    state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9"

    dataset = load_dataset("marsggbo/bigbench4switch64_patternand_pattern_predictor_gen")
    dataset.shuffle(seed=1234)
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'
    batch_id = 0

    for input_data, decode_id, pattern in process_schedule_dataset(dataset, tokenizer, schedule_size, num_experts_per_layer, top_n):
        predict_pattern = pattern.to(device)
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)

        input_list = truncate_input(input_ids, attention_mask, batch_size)
        
        # Schedule the first partition
        print(f"************* Batch {batch_id} *************")
        for token_id in range(max_new_tokens):
            print(f"Decode pos {token_id}")
            batch_index, _ = scheduler(predict_pattern[:, token_id+1].float(), cache_size, batch_size, 30, verbose=True)

        if batch_id == 8:
            break
        
        batch_id += 1
