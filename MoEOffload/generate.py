import torch
import time
import concurrent.futures
from MoEOffload.scheduler import scheduler, key_value_select_batch, key_value_select_merge
from transformers.modeling_outputs import MoEModelOutput

num_tasks = 1

def launch_predict(predictor,
                   input_ids,
                   decode_input_ids,
                   attention_mask,
                   pask_key_values,
                   encoder_outputs,
                   predict_stream):
    
    torch.cuda.nvtx.range_push(f"Predict")
    with torch.cuda.stream(predict_stream):
        predictor(input_ids=input_ids,
                decoder_input_ids=decode_input_ids,
                attention_mask=attention_mask,
                past_key_values=pask_key_values,
                encoder_outputs=encoder_outputs,
                use_cache=True)
    torch.cuda.nvtx.range_pop()


def fix_decode_generate(input_ids,
                        decode_ids,
                        attention_mask,
                        predict_pattern,
                        model,
                        predictor, 
                        executor,
                        cache_engine,
                        max_new_tokens=128,
                        past_key_values=None,
                        compute_stream=None,
                        predict_stream=None,
                        is_predict=True,
                        is_baseline=False, 
                        device=torch.device("cuda:0")):
    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = []
    past = past_key_values

    pattern = torch.zeros((24, 32), dtype=torch.int).to(device)
    decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
    encoder_outputs = None

    # print(f"Start inference")
    model.eval()  # Put model in evaluation mode
    predictor.eval()
    with torch.no_grad():  # Disable gradient calculation
        for step in range(1, max_new_tokens):
            torch.cuda.nvtx.range_push(f"Step {step}")
            if not is_baseline:
                torch.cuda.nvtx.range_push(f"Prefetch")
                cache_engine.prefetch(pattern)
                torch.cuda.nvtx.range_pop()
            
            if is_predict:
                futures = [executor.submit(launch_predict, predictor, input_ids, 
                                                    decoder_input_ids, attention_mask, 
                                                    past, encoder_outputs, predict_stream)]
    
            torch.cuda.nvtx.range_push(f"Compute")
            with torch.cuda.stream(compute_stream):
                outputs = model(input_ids=input_ids,
                                decoder_input_ids=decoder_input_ids,
                                attention_mask=attention_mask,
                                past_key_values=past,
                                encoder_outputs=encoder_outputs,
                                output_router_logits=True,
                                use_cache=True)  # use_cache允许模型返回past_key_values
            torch.cuda.nvtx.range_pop()
            # print(f"Step{step}: encoder-{outputs.encoder_router_logits[1][0].shape} decoder-{outputs.decoder_router_logits[1][0].shape}")
            
            # Select the next token based on the decode_id
            next_token = decode_ids[:, step]
            next_token = torch.unsqueeze(next_token, dim=-1).to(torch.int)

            # 应用temperature来调整预测分布
            generated_tokens.append(next_token)
            
            # decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            decoder_input_ids = next_token
            
            # Update the predict pattern
            if not is_baseline:
                pattern = predict_pattern[step]

            # Update Key-Value cache
            past = outputs.past_key_values

            # Update encoder outputs
            if encoder_outputs is None:
                encoder_outputs = MoEModelOutput(last_hidden_state=outputs.encoder_last_hidden_state,
                                                hidden_states=outputs.encoder_hidden_states,
                                                attentions=outputs.encoder_attentions,
                                                router_probs=outputs.encoder_router_logits)
            
            if is_predict:
                # Wait for all tasks to complete
                concurrent.futures.wait(futures)
            torch.cuda.nvtx.range_pop()

def schedule_generate(input_ids,
                    decode_ids,
                    attention_mask,
                    predict_pattern,
                    model,
                    predictor, 
                    executor,
                    cache_engine,
                    cache_size,
                    batch_size,
                    max_new_tokens=128,
                    past_key_values=None,
                    compute_stream=None,
                    predict_stream=None,
                    is_predict=True,
                    is_baseline=False, 
                    device=torch.device("cuda:0")):
    model.eval()

    decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
    encoder_outputs = None

    pattern = torch.zeros((24, 32), dtype=torch.int).to(device)
    cache_engine.prefetch(pattern)

    # Prefilling stage
    with torch.cuda.stream(compute_stream):
        outputs = model(input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        encoder_outputs=encoder_outputs,
                        output_router_logits=True,
                        use_cache=True)
    
    # Schedule the first partition
    batch_index, _ = scheduler(predict_pattern[:, 0].float(), cache_size, batch_size, 30)
    batch_key_value = key_value_select_batch(outputs.past_key_values, batch_index)
    num_minibatch = len(batch_index)

    # Decode stage
    duration = 0
    with torch.no_grad():
        for token_id in range(max_new_tokens):
            for i in range(num_minibatch):
                select_index = batch_index[i]

                encoder_outputs = MoEModelOutput(last_hidden_state=outputs.encoder_last_hidden_state[select_index],
                                                hidden_states=outputs.encoder_hidden_states,
                                                attentions=outputs.encoder_attentions,
                                                router_probs=outputs.encoder_router_logits)
                
                pattern = predict_pattern[select_index].sum(0)[token_id]
                key_values = batch_key_value[i]
                mask = attention_mask[select_index]
                decoder_input_ids = decode_ids[select_index, token_id]
                decoder_input_ids = torch.unsqueeze(decoder_input_ids, dim=-1)
                input = input_ids[select_index]

                torch.cuda.synchronize()
                start = time.time()

                cache_engine.prefetch(pattern)

                with torch.cuda.stream(compute_stream):
                    outputs = model(input_ids=input,
                                    decoder_input_ids=decoder_input_ids,
                                    attention_mask=mask,
                                    past_key_values=key_values,
                                    encoder_outputs=encoder_outputs,
                                    output_router_logits=True,
                                    use_cache=True)
                
                torch.cuda.synchronize()
                duration += time.time() - start

                batch_key_value[i] = outputs.past_key_values

                # Transform KV format and do batch schedule
                merge_key_value = key_value_select_merge(batch_key_value, batch_index)
                batch_index, _ = scheduler(predict_pattern[:, token_id+1].float(), cache_size, batch_size, 30)
                batch_key_value = key_value_select_batch(merge_key_value, batch_index)
    
    return duration