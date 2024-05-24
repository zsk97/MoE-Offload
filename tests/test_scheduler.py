import torch
import time

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

def update_clusters(data_similarity, indices):
    """更新簇分配，选择与质心最相似的点"""
    # 提取与质心相关的相似度
    cluster_similarities = data_similarity[:, indices]
    # 每个点被分配到最相似质心的簇中
    labels = torch.argmax(cluster_similarities, dim=1)
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

def kmeans_similarity(data_similarity, k, num_epochs=100):
    """执行基于相似度矩阵的K-means聚类"""
    n = data_similarity.size(0)
    indices = initialize_indices(n, k).cuda()
    labels = torch.zeros(n, dtype=torch.long, device='cuda')

    for epoch in range(num_epochs):
        new_labels = update_clusters(data_similarity, indices)
        if torch.equal(labels, new_labels):
            # print('break', epoch)
            break
        labels = new_labels
        indices, within_cluster_similarities = update_centroids(data_similarity, labels, k)
        # for i, idx in enumerate(indices):
        #     print(f"Epoch-{epoch} {idx} {within_cluster_similarities[i]}")

    # 收集每个簇的成员索引
    clusters = {i: (labels == i).nonzero(as_tuple=True)[0].cpu().numpy().tolist() for i in range(k)}
    return labels, indices, clusters
    # return list(clusters.items())

def sim_func(pattern_list):
    # 将矩阵展平
    patterns = torch.stack(pattern_list, dim=0)
    flat_patterns = patterns.view(patterns.size(0), -1)
    # 计算两两之间的Hamming距离
    dist = torch.cdist(flat_patterns, flat_patterns, p=0)
    # 将Hamming距离转换为相似度
    similarity = 1 - dist / flat_patterns.size(1)
    print(similarity)
    return similarity

def scheduler(pattern_list, num_epochs=30, k=8):
    data_similarity = sim_func(pattern_list)
    labels, centroids_indices, clusters = kmeans_similarity(data_similarity, k, num_epochs)
    indices_within_cluster = list(clusters.values())
    return indices_within_cluster

def balance_clusters(clusters, target_size):
    # 将簇的索引列表转换为数组以便操作
    cluster_sizes = {i: len(cluster) for i, cluster in clusters.items()}
    overfilled = {i: members for i, members in clusters.items() if len(members) > target_size}
    underfilled = {i: members for i, members in clusters.items() if len(members) < target_size}

    # 调整簇中的数据点分配
    adjusted_clusters = dict(clusters)  # 开始调整的副本
    transfer_list = []  # 存储需要移动的数据点

    # 收集过多的数据点
    for idx, members in overfilled.items():
        excess = len(members) - target_size
        transfer_list.extend((idx, member) for member in members[-excess:])
        adjusted_clusters[idx] = members[:-excess]  # 移除多余的成员

    # 将收集到的数据点重新分配到需要填充的簇中
    for idx, member in transfer_list:
        for u_idx, u_members in underfilled.items():
            if len(u_members) < target_size:
                adjusted_clusters[u_idx].append(member)
                underfilled[u_idx].append(member)
                if len(u_members) + 1 == target_size:
                    break

    # 确保调整后的簇大小正确
    assert all(len(members) == target_size for members in adjusted_clusters.values())

    return adjusted_clusters

def postprocess_clusters(indices_within_cluster, n, k):
    target_size = n // k
    clusters = {i: cluster for i, cluster in enumerate(indices_within_cluster)}
    balanced_clusters = balance_clusters(clusters, target_size)
    return list(balanced_clusters.values())

# 使用后处理调整簇的大小
# balanced_cluster_indices = postprocess_clusters(indices_within_cluster, n, k)
# for i, cluster in enumerate(balanced_cluster_indices):
#     print(f"Balanced Cluster {i} ({len(cluster)}): {cluster}")
# 示例用法
# n = 128  # 数据点的数量
# L = 6
# E = 32
# k = 2    # 簇的数量
# data = [torch.randint(0, 2, (L, E)).cuda().float() for _ in range(n)]

# for i in range(10):
#     indices_within_cluster = scheduler(data)

# start = time.time()
# indices_within_cluster = scheduler(data)
# end = time.time()
# print("Duration ", (end - start)*1000 )
# for i, cluster in enumerate(indices_within_cluster):
#     print(f"Cluster {i} ({len(cluster)}): {cluster}")
def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(torch.distributed.get_world_size())

def key_value_in_order(key_values, batch_size, num_batch):
    key_values_list = []

    num_layer = len(key_values)
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            kv_lists.append(torch.split(key_values[i][j], batch_size, dim=0))
        
        print("Length ", len(kv_lists[0]))
        
        results = []
        for tensors in zip(*kv_lists):
            results.append(tuple(tensors))
        
        print("Length tuple ", len(results[0]))
        break
def decode_in_order(model, 
                    cache_engine, 
                    encoder_outputs, 
                    decode_id, 
                    decode_pattern, 
                    attention_mask,
                    num_minibatch):
    batch_size = decode_id.shape[0] // num_minibatch

    compute_stream = torch.cuda.Stream()
    with torch.no_grad():
        for i in range(num_minibatch):
            # Create encoder outputs
            outputs = MoEModelOutput(last_hidden_state=encoder_outputs.encoder_last_hidden_state[i*batch_size:(i+1)*batch_size],
                                    hidden_states=encoder_outputs.encoder_hidden_states,
                                    attentions=encoder_outputs.encoder_attentions,
                                    router_probs=encoder_outputs.encoder_router_logits)

            pattern = decode_pattern[i*batch_size:(i+1)*batch_size].sum(0)[1]
            key_values = encoder_outputs.past_key_values[i*batch_size:(i+1)*batch_size]
            mask = attention_mask[i*batch_size:(i+1)*batch_size]
            decoder_input_ids = decode_id[i*batch_size:(i+1)*batch_size]

            cache_engine.prefetch(pattern)

            with torch.cuda.stream(compute_stream):
                outputs = model(input_ids=input_ids,
                                decoder_input_ids=decoder_input_ids,
                                attention_mask=mask,
                                past_key_values=key_values,
                                encoder_outputs=outputs,
                                output_router_logits=True,
                                use_cache=True)
            

if __name__ == "__main__":
    init_env()

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'

    state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9"

    dataset = load_dataset("marsggbo/bigbench4switch32_pattern_predictor_tmp")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'

    input_data, decode_id, batch_pattern, token_pattern = load_encoder(dataset, tokenizer, 128, 0)
    
    input_ids = input_data.input_ids.to(device)
    attention_mask = input_data.attention_mask.to(device)
    decoder_input_ids = decode_id.int().to(device)
    batch_pattern = batch_pattern.to(device)

    offload_model, cache_engine = build_offload_switch(offload_per_layer=16, state_path=state_path)
    offload_model = offload_model.bfloat16().to(device)

    pattern = torch.zeros((24, 32), dtype=torch.int).to(device)
    cache_engine.prefetch(pattern)

    # Prefilling
    outputs = offload_model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            encoder_outputs=None,
                            output_router_logits=True,
                            use_cache=True)
    
    print("Last hidden state shape ", outputs.encoder_last_hidden_state.shape)
    print("Past key value shape ", len(outputs.past_key_values))
    print("Type ", type(outputs.past_key_values))
    print("Key shape ", len(outputs.past_key_values[0]))
    print("Type ", type(outputs.past_key_values[0]))
    print("Shape ", outputs.past_key_values[0][0].shape)

    key_value_in_order(outputs.past_key_values, 8, 0)
    exit(0)

    decode_in_order(offload_model, cache_engine, outputs, decoder_input_ids, token_pattern, attention_mask, 4)
    