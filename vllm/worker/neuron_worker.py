"""A Neuron worker class."""
from typing import List, Tuple

import torch
import torch.distributed
import os
import datetime
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm.distributed.communication_op import (
    broadcast_tensor_dict)
from vllm.distributed.parallel_state import init_distributed_environment
from vllm.logger import init_logger

logger = init_logger(__name__)

class NeuronWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.is_driver_worker = False
        self.rank = None
        self.multi_node_inference = False
        if os.getenv("NEURON_MULTI_NODE", None) is not None:
            """Initialize the distributed environment."""
            self.multi_node_inference = True
            if os.getenv("NEURON_RANK_ID", None) is not None:
                self.rank = int(os.getenv("NEURON_RANK_ID"))
                if self.rank == 0:
                    self.is_driver_worker = True

        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank)

        self.model_runner = NeuronModelRunner(model_config, parallel_config,
                                              scheduler_config, device_config)
        
                
    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Neuron backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """

        # Different values are not tested.
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == self.scheduler_config.max_num_seqs

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata] = None,
    ) -> List[SamplerOutput]:
        if self.multi_node_inference:
            if self.is_driver_worker: # master node
                data = {
                    "seq_group_metadata_list": seq_group_metadata_list
                }
                broadcast_tensor_dict(data, src=0)
            else: # worker node
                data = broadcast_tensor_dict(src=0)
                seq_group_metadata_list = data["seq_group_metadata_list"]

        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []



        output = self.model_runner.execute_model(seq_group_metadata_list)

        # Neuron worker only supports single-step output. Wrap the output in a
        # list to conform to interface.
        return [output]

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError
