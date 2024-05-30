import torch
import time

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