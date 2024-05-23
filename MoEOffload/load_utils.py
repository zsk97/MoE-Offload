import torch

def process_dataset(dataset, tokenizer, batch_size):
    len_dataset = len(dataset['train'])
    num_batch = len_dataset // batch_size
    num_moe_layer = 6
    num_expert = 32
    num_layer = 24
    num_encoder_layer = 12

    for i in range(num_batch):
        prompts = []
        decode_id = []
        decode_pattern = []

        # Extract the batch info
        for j in range(batch_size):
            sample = dataset['train'][i*batch_size+j]
            prompts.append(sample['prompt_text'])
            decode_id.append(sample['decode_ids'])
            decode_pattern.append(sample['decode_pattern'])
        
        # Padding prompts
        input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

        decode_id = torch.Tensor(decode_id)
        decode_length = decode_id.shape[-1]

        # Deal with pattner
        decode_pattern = torch.Tensor(decode_pattern)
        decode_pattern = decode_pattern.permute((2, 1, 0))
        
        # Switch Transformer use MoE in non-adjacent layer
        # Currently, we only have pattern for decoder

        pattern = torch.zeros((decode_length, num_layer, num_expert), dtype=torch.int)
        for token_id in range(decode_length):
            for j in range(num_moe_layer):
                decode_layer_id = num_encoder_layer + j*2 + 1
                batch_pattern = decode_pattern[token_id][j].to(int).flatten().unique().tolist()
                pattern[token_id][decode_layer_id][batch_pattern] = 1

        yield input_data, decode_id, pattern