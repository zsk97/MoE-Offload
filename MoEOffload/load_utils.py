import torch
import logging

def process_dataset(dataset, tokenizer, batch_size, num_expert, top_n=0):
    len_dataset = len(dataset['train'])
    num_batch = len_dataset // batch_size
    num_encoder_layer = 12

    if top_n == 0:
        logging.info("Process real decode pattern")
    else:
        logging.info(f"Process top {top_n} pattern")

    for i in range(num_batch):
        prompts = []
        decode_id = []
        decode_pattern = []
        predict_pattern = []

        # Extract the batch info
        for j in range(batch_size):
            sample = dataset['train'][i*batch_size+j]
            prompts.append(sample['prompt_text'])
            decode_id.append(sample['decode_ids'])
            decode_pattern.append(sample['decode_pattern'])
            predict_pattern.append(sample['predictor_pattern'])
        
        # Padding prompts
        input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

        decode_id = torch.Tensor(decode_id).long()
        decode_length = decode_id.shape[-1]

        # Deal with pattner
        decode_pattern = torch.Tensor(decode_pattern).long()
        decode_pattern = decode_pattern.permute((0, 2, 1))
        predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, top3_indices)
        
        pattern = None
        # Switch Transformer use MoE in non-adjacent layer
        # Currently, we only have pattern for decoder

        # We use real pattern
        if top_n == 0:
            onehot_decode_pattern = torch.nn.functional.one_hot(
                decode_pattern, num_classes=num_expert
            )
            batch_decode_pattern_real = onehot_decode_pattern.sum(0) # (seq_len, num_moe_layers, num_expert))
            batch_encode_pattern = torch.zeros((decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            batch_decode_pattern = torch.zeros((decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            indices = list(range(1, num_encoder_layer, 2))
            batch_decode_pattern[:, indices, :] = batch_decode_pattern_real
            pattern = torch.cat((batch_encode_pattern, batch_decode_pattern), dim=1)
        else:
            top_predict_pattern = predict_pattern[..., :top_n]
            onehot_predict_pattern = torch.nn.functional.one_hot(
                top_predict_pattern, num_classes=num_expert
            )
            batch_predict_pattern = onehot_predict_pattern.sum(-2).sum(0) # Sum along batch size and top n indices
            batch_encode_pattern = torch.zeros((decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            batch_decode_pattern = torch.zeros((decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            indices = list(range(1, num_encoder_layer, 2))
            batch_decode_pattern[:, indices, :] = batch_predict_pattern
            pattern = torch.cat((batch_encode_pattern, batch_decode_pattern), dim=1)

        yield input_data, decode_id, pattern

def process_schedule_dataset(dataset, tokenizer, batch_size, num_expert, top_n=0):
    len_dataset = len(dataset['train'])
    num_batch = len_dataset // batch_size
    num_encoder_layer = 12

    if top_n == 0:
        logging.info("Process real decode pattern")
    else:
        logging.info(f"Process top {top_n} pattern")

    for i in range(num_batch):
        prompts = []
        decode_id = []
        decode_pattern = []
        predict_pattern = []

        # Extract the batch info
        for j in range(batch_size):
            sample = dataset['train'][i*batch_size+j]
            prompts.append(sample['prompt_text'])
            decode_id.append(sample['decode_ids'])
            decode_pattern.append(sample['decode_pattern'])
            predict_pattern.append(sample['predictor_pattern'])
        
        # Padding prompts
        input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

        decode_id = torch.Tensor(decode_id)
        decode_length = decode_id.shape[-1]

        # Deal with pattner
        decode_pattern = torch.Tensor(decode_pattern).long()
        decode_pattern = decode_pattern.permute((0, 2, 1))
        predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, top3_indices)

        pattern = None
        # Switch Transformer use MoE in non-adjacent layer
        # Currently, we only have pattern for decoder

        # We use real pattern
        if top_n == 0:
            onehot_decode_pattern = torch.nn.functional.one_hot(
                decode_pattern, num_classes=num_expert
            )
            token_decode_pattern_real = onehot_decode_pattern # (bs, seq_len, num_moe_layers, num_expert))
            token_encode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            token_decode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            indices = list(range(1, num_encoder_layer, 2))
            token_decode_pattern[:, :, indices, :] = token_decode_pattern_real
            pattern = torch.cat((token_encode_pattern, token_decode_pattern), dim=2)
        else:
            top_predict_pattern = predict_pattern[..., :top_n]
            onehot_predict_pattern = torch.nn.functional.one_hot(
                top_predict_pattern, num_classes=num_expert
            )
            token_predict_pattern = onehot_predict_pattern.sum(-2) # Sum along top n indices
            token_encode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            token_decode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
            indices = list(range(1, num_encoder_layer, 2))
            token_decode_pattern[:, :, indices, :] = token_predict_pattern
            pattern = torch.cat((token_encode_pattern, token_decode_pattern), dim=2)
    
        yield input_data, decode_id, pattern
    

def load_encoder(dataset, tokenizer, batch_size, batch_idx):
    len_dataset = len(dataset['train'])
    num_batch = len_dataset // batch_size
    num_moe_layer = 6
    num_expert = 32
    num_layer = 24
    num_encoder_layer = 12

    prompts = []
    decode_id = []
    decode_pattern = []
    i = batch_idx

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

    decode_pattern = torch.Tensor(decode_pattern) # (128, 6, 8)

    print("Decode id shape ", decode_id.shape)
    print("Decode pattern shape ", decode_pattern.shape)
    decode_pattern = decode_pattern.permute((0, 2, 1)) # (128, 8, 6)

    token_pattern = torch.zeros((batch_size, decode_length, num_layer, num_expert), dtype=torch.int)
    for batch_id in range(batch_size):
        for token_id in range(decode_length):
            for j in range(num_moe_layer):
                decode_layer_id = num_encoder_layer + j*2 + 1
                batch_pattern = decode_pattern[batch_id][token_id][j].to(int)
                token_pattern[batch_id][token_id][decode_layer_id][batch_pattern] = 1
    
    decode_pattern = decode_pattern.permute((1, 2, 0))
    pattern = torch.zeros((decode_length, num_layer, num_expert), dtype=torch.int)
    for token_id in range(decode_length):
        for j in range(num_moe_layer):
            decode_layer_id = num_encoder_layer + j*2 + 1
            batch_pattern = decode_pattern[token_id][j].to(int).flatten().unique().tolist()
            pattern[token_id][decode_layer_id][batch_pattern] = 1
    
    return input_data, decode_id, pattern, token_pattern