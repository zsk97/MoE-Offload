import torch
import concurrent.futures
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