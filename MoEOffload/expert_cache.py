from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
from .expert_wrapper import MixtralExpertWrapper

import torch
from torch import nn

ExpertUID = Any

@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    gpu_index: int
    cpu_index: int


@dataclass
class EvictionGroupInfo:
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            return info  # least recently used
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        # print("Info to load id", info_to_load.uid)
        # print("Info to evict id", info_to_evict.uid)
        # print("*************")
        # print(self.offloaded_infos)
        # print("*************")
        # print(self.main_infos)
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")
        
    def expert_in_gpu(self):
        experts = []
        for uid, info in self.main_infos.items():
            experts.append(info)
        return experts
    
    def uid_in_gpu(self):
        uid_list = []
        for uid, info in self.main_infos.items():
            uid_list.append(uid)
        return uid_list


class ExpertCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, num_layer: int = None):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)])
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

        # 添加命中率统计变量
        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, cpu_index=i, gpu_index=-1)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, cpu_index=i, gpu_index=-1)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.cpu_index] for info in pre_loaded_infos])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load = deque([info for info in infos if info.offloaded])
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            window_size = min(len(self.device_expert_buffers) - 1,
                              len(eviction_group.main_infos),
                              len(infos_to_load))
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._swap(info_to_load, eviction_group.choose_expert_to_evict()))

            for info in infos:
                self.num_accesses += 1
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    self.num_hits += 1
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    self.num_accesses += 1
                    yield (info.uid, experts_in_loading.popleft())
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._swap(info_to_load, eviction_group.choose_expert_to_evict()))
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.cpu_index], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.cpu_index].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[info_to_evict.cpu_index])
        self.main_modules[info_to_evict.cpu_index] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[info_to_load.cpu_index])
        self.offloaded_storages[info_to_load.cpu_index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.cpu_index] = info_to_load
        self.offloaded_infos[info_to_load.cpu_index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.cpu_index, info_to_load.cpu_index = info_to_load.cpu_index, info_to_evict.cpu_index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        return device_expert_buffer

    def prefetch(
            self,
            pattern_matrix,
    ) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        Pre-fetches experts based on a provided activation matrix for future layers.

        Args:
            pattern_matrix (torch.Tensor): A matrix indicating the activation state of experts across layers.

        Returns:
            Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]: Iterator for pre-fetched expert modules.
        """
        
        num_layers, num_experts = pattern_matrix.shape

        # 1. 统计出当前成 preloaded 和 offloaded 状态的 infos，根据 pattern_matrix 记录每个 expert 的状态：
        #  1) 无需操作：当 pattern_matrix[layer_id, expert_id] == 1 时,且 info.offloaded=False，该 expert 需要被用到，且已经 preload在 GPU 上
        #  1) 无需操作：当 pattern_matrix[layer_id, expert_id] == 0 时,且 info.offloaded=True，该 expert  不需要被用到，且已经 offload 在 GPU 上
        #  2) 需要 offload：当 pattern_matrix[layer_id, expert_id] == 0 时,且 info.offloaded=False，该 expert 不需要被用到，但已经 preload 到 GPU 上
        #  3）需要 preload: 当 pattern_matrix[layer_id, expert_id] == 1 时,且 info.offloaded=True，该 expert 需要被用到，但当前被 offload 到 CPU 上
        
        num_failed_preload = 0.
        for layer_id in range(num_layers):
            cpu2gpu_infos = []
            gpu2cpu_infos = []
            for expert_id in range(num_experts):
                uid: ExpertUID = (layer_id, expert_id)
                info = self.registered_experts.get(uid)
                
                # Skip if expert info is not found
                if info is None:
                    continue
                required_on_gpu = pattern_matrix[layer_id, expert_id] == 1
                if required_on_gpu and info.offloaded:
                    cpu2gpu_infos.append(info)
                elif not required_on_gpu and not info.offloaded:
                    gpu2cpu_infos.append(info)

            # Perform swaps
            while cpu2gpu_infos and gpu2cpu_infos:
                info_to_load = cpu2gpu_infos.pop()
                info_to_evict = gpu2cpu_infos.pop()
                # print(f"Swaping {info_to_load.uid}(cpu) to {info_to_evict.uid}(gpu)")
                self._swap(info_to_load, info_to_evict)
            
            # Todo: 支持在不同层之间的 expert 互相替换。
            # Todo: 因为每层的 expert 激活数量可能不一致，比如第一层只激活了 2 个，第二个需要激活 8 个（默认激活 4 个，offload4 个），那么已经 offload 的 4 个可以挪到第一层中去
            # Check remaining unprocessed experts due to imbalance in requirements
            # if len(cpu2gpu_infos) > 0:
            #     num_failed_preload += len(cpu2gpu_infos)
            #     failed_uids = [e.uid for e in cpu2gpu_infos]
            #     print(f"Layer{layer_id} has {len(cpu2gpu_infos)} experts {failed_uids} that cannot be preloaded.")

    def get_hit_rate(self) -> float:
        """计算命中率"""
        if self.num_accesses == 0:
            return 0.0
        return self.num_hits / self.num_accesses

    def get_miss_rate(self) -> float:
        """计算缺失率"""
        if self.num_accesses == 0:
            return 0.0
        return self.num_misses / self.num_accesses

    def reset_stats(self):
        """重置统计信息"""
        self.num_hits = 0
        self.num_misses = 0 
        self.num_accesses = 0

class ExpertCacheV1(object):
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, num_layer: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        print("Create main module list")
        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        print("Create offload storage list")
        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)])
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

        print("Group info ", self.group_infos)

        self.prefetch_stream = torch.cuda.Stream()
        self.ondemand_stream = torch.cuda.Stream(priority=-1)

        self.num_layer = num_layer
        print("Number layer ", num_layer)
        self.event_queue = [None] * self.num_layer
        # 添加命中率统计变量
        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)
    
    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size, f"Storage size {len(storage)} != module size{self.module_size}"

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, gpu_index=i, cpu_index=-1)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    break  # done allocating; found spot on device
            
            # We always keep a copy in CPU
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    self.registered_experts[uid].cpu_index = i
                    self.offloaded_infos[i] = info
                    return

        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, gpu_index=-1, cpu_index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return
        raise ValueError("Cache is full")
    
    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        assert info_to_load.eviction_group == info_to_evict.eviction_group

        self.main_modules[info_to_evict.gpu_index].storage.copy_(self.offloaded_storages[info_to_load.cpu_index], non_blocking=True)
        self.main_infos[info_to_evict.gpu_index] = info_to_load
        
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_load.gpu_index = info_to_evict.gpu_index
        info_to_evict.gpu_index = -1
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        
    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)
        
        try:
            self.active = True
            
            # Save pre-loaded experts before swap
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.gpu_index] for info in pre_loaded_infos])

            infos_to_load = deque([info for info in infos if info.offloaded])

            # print(f" {len(infos_to_load)} experts should be loaded")
            event_to_sync = dict()
            for info in infos_to_load:
                event_to_sync[info.uid] = torch.cuda.Event()

            # Check current available position
            pre_loaded_uid = set([info.uid for info in pre_loaded_infos])
            exist_info = eviction_group.expert_in_gpu()

            evict_experts_info = deque()
            for expert_info in exist_info:
                if expert_info.uid not in pre_loaded_uid:
                    evict_experts_info.append(expert_info)
            
            # print("========== Evict expert ==========")
            # print(evict_experts_info)
            # print("========== Evict Load ==========")
            # print(infos_to_load)
            # If there are still available space, we should launch
            # prefetch here before the computation 
            idx = 0
            while len(evict_experts_info) > 0:
                with torch.cuda.stream(self.ondemand_stream):
                    if idx >= len(infos_to_load):
                        break
                    self._swap(infos_to_load[idx], evict_experts_info.popleft())
                    event_to_sync[infos_to_load[idx].uid].record()
                    idx += 1

            # Record the finished computation expert
            finish_experts_info = deque()
            for info in infos:
                self.num_accesses += 1
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    self.num_hits += 1
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_to_load) > 0:
                    self.num_misses += 1
                    # Syn the copy and return expert
                    event_to_sync[info.uid].synchronize()
                    expert = self.main_modules[info.gpu_index]

                    infos_to_load.popleft()
                    idx -= 1
                    yield (info.uid, expert)
                
                # Swap out the finished expert 
                # TODO: Ensure the computation finished by using cuda event
                finish_experts_info.append(info)
                curr_stream = torch.cuda.current_stream()
                curr_stream.synchronize()

                # Launch the request for next copy
                if idx < len(infos_to_load):
                    with torch.cuda.stream(self.ondemand_stream):
                        self._swap(infos_to_load[idx], finish_experts_info[0])
                        event_to_sync[infos_to_load[idx].uid].record()
                    finish_experts_info.popleft()
                    idx += 1
        finally:
            self.active = False
    
    def prefetch(self, pattern: torch.Tensor):
        num_layers, num_experts = pattern.shape

        experts = torch.nonzero(pattern, as_tuple=False).cpu().numpy().tolist()
        required_experts = set()
        for expert in experts:
            required_experts.add(tuple(expert))

        with torch.cuda.stream(self.prefetch_stream):
            # Only work for switch transformer
            for layer_id in range(1, num_layers, 2):
                cpu2gpu_infos = []
                eviction_group = None
                
                for expert_id in range(num_experts):
                    uid: ExpertUID = (layer_id, expert_id)
                    info = self.registered_experts.get(uid)

                    assert info is not None, f"Unregonized Expert {uid}!"
                    required_on_gpu = False
                    if uid in required_experts:
                        required_on_gpu = True

                    if required_on_gpu and info.offloaded:
                        cpu2gpu_infos.append(info)
                    
                    if eviction_group is None:
                        eviction_group = self.group_infos[info.eviction_group]
                
                # Get evict expert
                expert_in_gpu = eviction_group.expert_in_gpu()
                evict_experts = []
                for expert_info in expert_in_gpu:
                    if expert_info.uid not in required_experts:
                        evict_experts.append(expert_info)


                while cpu2gpu_infos and evict_experts:
                    info_to_load = cpu2gpu_infos.pop()
                    info_to_evict = evict_experts.pop()

                    self._swap(info_to_load, info_to_evict)

                self.event_queue[layer_id] = torch.cuda.Event()
                self.event_queue[layer_id].record()
    
    def sync_layer(self, layer_id):
        self.event_queue[layer_id].synchronize()
    
    def check_main_module(self, pattern):
        for i in range(1, self.num_layer, 2):
            print("Layer ID ", i)
            eviction_group = self.group_infos[i]
            expert_list = eviction_group.expert_in_gpu()
            for expert in expert_list:
                print(expert.uid)
            print("******************")
            print(torch.nonzero(pattern[i], as_tuple=False))

    def get_hit_rate(self) -> float:
        """计算命中率"""
        if self.num_accesses == 0:
            return 0.0
        return self.num_hits / self.num_accesses

    def get_miss_rate(self) -> float:
        """计算缺失率"""
        if self.num_accesses == 0:
            return 0.0
        return self.num_misses / self.num_accesses

    def reset_stats(self):
        """重置统计信息"""
        self.num_hits = 0
        self.num_misses = 0 
        self.num_accesses = 0