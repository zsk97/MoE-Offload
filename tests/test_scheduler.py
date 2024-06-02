import torch
import time
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import MoEModelOutput

from accessory.util import misc
import fairscale.nn.model_parallel.initialize as fs_init

from MoEOffload.load_utils import load_encoder
from MoEOffload.load_utils import process_dataset
from MoEOffload.generate import fix_decode_generate
from MoEOffload.build_model import build_offload_switch
from MoEOffload.args import parse_args

def initialize_indices(n, k):
    """随机初始化质心的索引"""
    indices = torch.randperm(n)[:k]
    return indices

def update_clusters(data_similarity, indices, n, k, is_balanced=False):
    """更新簇分配，选择与质心最相似的点"""
    # 提取与质心相关的相似度
    cluster_similarities = data_similarity[:, indices]
    # 每个点被分配到最相似质心的簇中
    labels = torch.argmax(cluster_similarities, dim=1)

    if is_balanced:
        # 每个簇的目标大小
        target_cluster_size = n // k
        cluster_sizes = torch.bincount(labels, minlength=k)

        # 首先识别并处理超出目标大小的簇
        for i in range(k):
            if cluster_sizes[i] > target_cluster_size:
                excess_amount = cluster_sizes[i] - target_cluster_size
                cluster_indices = (labels == i).nonzero(as_tuple=True)[0]
                # 根据与质心的相似度排序，保留前 target_cluster_size 个最相似的点
                similarities = cluster_similarities[cluster_indices, i]
                _, sorted_indices = similarities.sort(descending=True)
                # 设置超出部分的点标签为未分配（-1或可以用一个特殊值）
                labels[cluster_indices[sorted_indices[target_cluster_size:]]] = -1

        # 处理数量少于目标大小的簇
        unassigned_indices = (labels == -1).nonzero(as_tuple=True)[0]
        unassigned_similarities = data_similarity[unassigned_indices, :]
        for i in range(k):
            if cluster_sizes[i] < target_cluster_size:
                needed_amount = target_cluster_size - cluster_sizes[i]
                # 选择未分配点中相对于当前质心相似度最高的点
                top_candidates = torch.topk(unassigned_similarities[:, i], needed_amount).indices
                labels[unassigned_indices[top_candidates]] = i
                # 更新未分配点列表
                unassigned_indices = (labels == -1).nonzero(as_tuple=True)[0]
                unassigned_similarities = data_similarity[unassigned_indices, :]

    return labels

def update_centroids(data_similarity, labels, k):
    """根据新的簇分配更新质心"""
    new_indices = torch.zeros(k, dtype=torch.long, device='cuda')
    within_cluster_similarities = []
    for i in range(k):
        # 找到属于同一簇的所有点的相似度总和
        within_cluster_similarity = data_similarity[labels == i][:, labels == i].sum(dim=1)
        # 选择使内部相似度最大化的点作为新质心
        new_indices[i] = (labels == i).nonzero()[within_cluster_similarity.argmax()]
        within_cluster_similarities.append(within_cluster_similarity.max())
    return new_indices, within_cluster_similarities

def kmeans_similarity(data_similarity, k, num_epochs=100, is_balanced=True):
    """执行基于相似度矩阵的K-means聚类"""
    n = data_similarity.size(0)
    indices = initialize_indices(n, k).cuda()
    labels = torch.zeros(n, dtype=torch.long, device='cuda')

    for epoch in range(num_epochs):
        new_labels = update_clusters(data_similarity, indices, n, k, is_balanced)
        if len(set(new_labels.cpu().tolist()))<k:
            # in case some clusters have only one data point except centroids point
            # re-initialize the centroids to avoid empty clusters
            indices = initialize_indices(n, k).cuda()
            continue
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
        indices, within_cluster_similarities = update_centroids(data_similarity, labels, k)
        # for i, idx in enumerate(indices):
        #     print(f"Epoch-{epoch} {idx} {within_cluster_similarities[i]}")

    # 收集每个簇的成员索引
    clusters = {i: (labels == i).nonzero(as_tuple=True)[0].cpu().numpy().tolist() for i in range(k)}
    # print([len(x) for x in clusters.values()])
    assert sum([len(x) for x in clusters.values()])==n
    return labels, indices, clusters

def sim_func(pattern_list):
    # 将矩阵展平
    # patterns = torch.stack(pattern_list, dim=0)
    flat_patterns = pattern_list.view(pattern_list.size(0), -1)
    # 计算两两之间的Hamming距离
    dist = torch.cdist(flat_patterns, flat_patterns, p=0)
    # 将Hamming距离转换为相似度
    similarity = 1 - dist / flat_patterns.size(1)
    return similarity

def scheduler(pattern_list, cache_size, batch_size, num_epochs=30, is_balanced=True, verbose=False):
    k = pattern_list.shape[0] // batch_size
    data_similarity = sim_func(pattern_list)
    labels, centroids_indices, clusters = kmeans_similarity(data_similarity, k, num_epochs, is_balanced)
    indices_within_cluster = list(clusters.values())
    # print("Output index length ", len(indices_within_cluster))

    on_demand_expert_schedule = []
    for i, cluster in enumerate(indices_within_cluster):
        cluster_pattern_list = [pattern_list[idx] for idx in cluster]
        cluster_pattern = torch.stack(cluster_pattern_list, dim=0).sum(0)
        num_activated_experts_per_layer = (cluster_pattern>0).sum(-1).cpu()
        num_activated_experts_per_layer -= cache_size
        on_demand_expert_schedule.append(torch.sum(num_activated_experts_per_layer[num_activated_experts_per_layer > 0]))
    if verbose:
        print("[Schedule] Number ondemand load ", sum(on_demand_expert_schedule), on_demand_expert_schedule)

    on_demand_expert_sequential = []
    sorted_cluster_indices = list(range(pattern_list.shape[0]))
    for batch_id in range(len(indices_within_cluster)):
        cluster = sorted_cluster_indices[batch_id*batch_size:(batch_id+1)*batch_size]
        cluster_pattern_list = [pattern_list[idx] for idx in cluster]
        cluster_pattern = torch.stack(cluster_pattern_list, dim=0).sum(0)
        num_activated_experts_per_layer = (cluster_pattern>0).sum(-1).cpu()
        num_activated_experts_per_layer -= cache_size
        on_demand_expert_sequential.append(torch.sum(num_activated_experts_per_layer[num_activated_experts_per_layer > 0]))
    if verbose:
        print("[Sorted] Number ondemand load ", sum(on_demand_expert_sequential), on_demand_expert_sequential)

    return indices_within_cluster, (sum(on_demand_expert_schedule).item(), sum(on_demand_expert_sequential).item())


def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(torch.distributed.get_world_size())

def key_value_in_order(key_values, batch_size):
    key_values_list = []

    num_layer = len(key_values)
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            kv_lists.append(torch.split(key_values[i][j], batch_size, dim=0))
        
        results = []
        for tensors in zip(*kv_lists):
            results.append(tuple(tensors))
        
        key_values_list.append(results)
    
    final_results = []
    for elements in zip(*key_values_list):
        final_results.append(tuple(elements))
    
    return final_results

def key_value_select_batch(key_values, batch_idx):
    key_values_list = []

    num_layer = len(key_values)
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            subtensors = [key_values[i][j][index] for index in batch_idx]
            kv_lists.append(subtensors)
        
        results = []
        for tensors in zip(*kv_lists):
            results.append(tuple(tensors))

        key_values_list.append(results)
    
    final_results = []
    for elements in zip(*key_values_list):
        final_results.append(tuple(elements))

    return final_results

def key_value_select_merge(key_value_list, batch_idx):
    num_batch = len(key_value_list)
    num_layer = len(key_value_list[0])

    batch_size, *kv_shape_self = key_value_list[0][0][0].shape
    _, *kv_shape_cross = key_value_list[0][0][2].shape
    kv_type = key_value_list[0][0][0].dtype
    device = key_value_list[0][0][0].device

    merge_kv_list = []
    kv_tensor = None
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            if j < 2:  
                kv_tensor = torch.zeros((num_batch * batch_size, *kv_shape_self), dtype=kv_type, device=device)
                # print("KV tensor shape ", kv_tensor.shape)
            else:
                kv_tensor = torch.zeros((num_batch * batch_size, *kv_shape_cross), dtype=kv_type, device=device)
                # print("KV tensor shape ", kv_tensor.shape)
            for batch_id in range(num_batch):
                # print(f"Batch {batch_id} Layer {i} pos {j}")
                # print("Fill tensor shape ", key_value_list[batch_id][i][j].shape)
                kv_tensor[batch_idx[batch_id], ...] = key_value_list[batch_id][i][j]
            kv_lists.append(kv_tensor)
        
        merge_kv_list.append(tuple(kv_lists))

    return tuple(merge_kv_list)


def decode_in_order(model, 
                    cache_engine, 
                    encoder_outputs, 
                    input_id,
                    decode_id, 
                    decode_pattern, 
                    attention_mask,
                    batch_key_value,
                    max_new_tokens,
                    num_minibatch):
    batch_size = decode_id.shape[0] // num_minibatch

    compute_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for token_id in range(max_new_tokens):
            for i in range(num_minibatch):
                # Create encoder outputs
                torch.cuda.nvtx.range_push(f"Batch {i}")
                outputs = MoEModelOutput(last_hidden_state=encoder_outputs.encoder_last_hidden_state[i*batch_size:(i+1)*batch_size],
                                        hidden_states=encoder_outputs.encoder_hidden_states,
                                        attentions=encoder_outputs.encoder_attentions,
                                        router_probs=encoder_outputs.encoder_router_logits)

                pattern = decode_pattern[i*batch_size:(i+1)*batch_size].sum(0)[token_id]
                key_values = batch_key_value[i]
                mask = attention_mask[i*batch_size:(i+1)*batch_size]
                decoder_input_ids = decode_id[i*batch_size:(i+1)*batch_size, token_id]
                decoder_input_ids = torch.unsqueeze(decoder_input_ids, dim=-1).to(torch.long)
                input = input_id[i*batch_size:(i+1)*batch_size]

                cache_engine.prefetch(pattern)

                with torch.cuda.stream(compute_stream):
                    outputs = model(input_ids=input,
                                    decoder_input_ids=decoder_input_ids,
                                    attention_mask=mask,
                                    past_key_values=key_values,
                                    encoder_outputs=outputs,
                                    output_router_logits=True,
                                    use_cache=True)
                torch.cuda.nvtx.range_pop()

                batch_key_value[i] = outputs.past_key_values
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")

def decode_in_select_batch(model, 
                        cache_engine, 
                        encoder_outputs,
                        input_ids, 
                        decode_id, 
                        decode_pattern, 
                        attention_mask,
                        batch_key_value,
                        max_new_tokens,
                        batch_size,
                        cache_size,
                        batch_index):
    num_minibatch = len(batch_index)

    compute_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    duration = 0
    with torch.no_grad():
        for token_id in range(max_new_tokens):
            for i in range(num_minibatch):
                select_index = batch_index[i]
                # Create encoder outputs
                outputs = MoEModelOutput(last_hidden_state=encoder_outputs.encoder_last_hidden_state[select_index],
                                        hidden_states=encoder_outputs.encoder_hidden_states,
                                        attentions=encoder_outputs.encoder_attentions,
                                        router_probs=encoder_outputs.encoder_router_logits)

                pattern = decode_pattern[select_index].sum(0)[token_id]
                key_values = batch_key_value[i]
                mask = attention_mask[select_index]
                decoder_input_ids = decode_id[select_index, token_id]
                decoder_input_ids = torch.unsqueeze(decoder_input_ids, dim=-1).to(torch.long)
                input = input_ids[select_index]

                torch.cuda.synchronize()
                start = time.time()

                cache_engine.prefetch(pattern)

                with torch.cuda.stream(compute_stream):
                    outputs = model(input_ids=input,
                                    decoder_input_ids=decoder_input_ids,
                                    attention_mask=mask,
                                    past_key_values=key_values,
                                    encoder_outputs=outputs,
                                    output_router_logits=True,
                                    use_cache=True)
                
                torch.cuda.synchronize()
                duration += time.time() - start

                batch_key_value[i] = outputs.past_key_values
            
            # Collect the key value cache 
            torch.cuda.synchronize()
            start = time.time()
            merge_key_value = key_value_select_merge(batch_key_value, batch_index)
            torch.cuda.synchronize()
            end = time.time()

            print("KV merge time ", end - start)
            # reschedule
            # print("pattern list ", decode_pattern[:, token_id+1].shape)
            torch.cuda.synchronize()
            start = time.time()
            batch_index, _ = scheduler(decode_pattern[:, token_id+1].float(), cache_size, batch_size, 30)
            torch.cuda.synchronize()
            end = time.time()
            print("Schedule time ", end - start)
            # print("batch index length ", len(batch_index))

            torch.cuda.synchronize()
            start = time.time()
            batch_key_value = key_value_select_batch(merge_key_value, batch_index)
            torch.cuda.synchronize()
            end = time.time()
            print("KV select time ", end - start)
            # print("batch key value length ", len(batch_key_value))

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Compute time: {duration * 1000} ms")
    print(f"Elapsed time: {elapsed_time_ms} ms")

if __name__ == "__main__":
    init_env()

    in_order = True
    cache_size = 8
    batch_size = 32
    total_batch_size = 128
    total_batch_id = 1
    max_new_tokens = 7

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'

    state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9"

    dataset = load_dataset("marsggbo/bigbench4switch64_patternand_pattern_predictor_gen")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'

    input_data, decode_id, batch_pattern, token_pattern = load_encoder(dataset, tokenizer, total_batch_size, total_batch_id)
    
    input_ids = input_data.input_ids.to(device)
    attention_mask = input_data.attention_mask.to(device)
    decoder_input_ids = decode_id.int().to(device)
    batch_pattern = batch_pattern.to(device)

    offload_model, cache_engine = build_offload_switch(offload_per_layer=16, state_path=state_path, model_name="google/switch-base-32", is_baseline=False, is_profile=True)
    offload_model = offload_model.bfloat16().to(device)

    pattern = torch.zeros((24, 32), dtype=torch.int).to(device)
    cache_engine.prefetch(pattern)

    print("Token pattern shape ", token_pattern.shape)
    print("Shape of batch pattern ", batch_pattern.shape)
    print("input id shape ", input_ids.shape)

    # balanced_cluster_indices = postprocess_clusters(indices_within_cluster, 128, 4)

    # Prefilling stage
    outputs = offload_model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            encoder_outputs=None,
                            output_router_logits=True,
                            use_cache=True)
    
    if in_order:
        batch_key_value = key_value_in_order(outputs.past_key_values, batch_size)

        torch.cuda.cudart().cudaProfilerStart()
        decode_in_order(offload_model, cache_engine, outputs, input_ids, decoder_input_ids, token_pattern, attention_mask, batch_key_value, max_new_tokens, 4)
        torch.cuda.cudart().cudaProfilerStop()
    else:
        indices_within_cluster, _ = scheduler(token_pattern[:, 0].float().to(device), cache_size, batch_size, 30)
        print("Origin KV cache size ", outputs.past_key_values[0][0].shape)
        print("Origin probelm size ", outputs.past_key_values[0][2].shape)
        batch_key_value = key_value_select_batch(outputs.past_key_values, indices_within_cluster)
        print("After schedule KV cache size ", batch_key_value[0][0][0].shape)
        print("After scheduleproblem size ", batch_key_value[0][0][2].shape)
        decode_in_select_batch(offload_model, cache_engine, outputs, input_ids, decoder_input_ids, token_pattern.cuda(), attention_mask, batch_key_value, max_new_tokens, batch_size, cache_size, indices_within_cluster)


