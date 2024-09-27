"""A Neuron worker class."""
import enum
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.neuronx_distributed_model_runner import NeuronxDistributedModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)
from vllm.utils import is_transformers_neuronx, is_neuronx_distributed_inference


class NeuronFramework(enum.Enum):
    TRANSFORMERS_NEURONX = "transformers-neuronx"
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"

class NeuronWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        neuron_framework = self._get_neuron_framework_to_use()

        if neuron_framework == NeuronFramework.TRANSFORMERS_NEURONX:
            self.model_runner: NeuronModelRunner = NeuronModelRunner(
                model_config, parallel_config, scheduler_config, device_config)
        elif neuron_framework == NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE:
            self.model_runner: NeuronxDistributedModelRunner = NeuronxDistributedModelRunner(
                model_config, parallel_config, scheduler_config, device_config)
        else:
            raise NotImplementedError(
                f"Specified framework as {os.environ.get('VLLM_NEURON_FRAMEWORK')}," +
                " Only transformers-neuronx/neuronx-distributed-inference framework is supported")
        self.is_driver_worker = True

    def _get_neuron_framework_to_use(self):
        """
        Return the specified framework if the corresponding installations are available.
        If no framework is specified, then use transformers-neuronx by default, if unavailable
        then check and switch to neuronx-distributed-inference.
        """
        transformers_neuronx_installed = is_transformers_neuronx()
        neuronx_distributed_inference_installed = is_neuronx_distributed_inference()
        specified_framework = os.environ.get("VLLM_NEURON_FRAMEWORK")
        if specified_framework == NeuronFramework.TRANSFORMERS_NEURONX.value and transformers_neuronx_installed:
            return NeuronFramework.TRANSFORMERS_NEURONX
        elif specified_framework == NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE.value and neuronx_distributed_inference_installed:
            return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
        elif specified_framework is None and transformers_neuronx_installed:
            return NeuronFramework.TRANSFORMERS_NEURONX
        elif specified_framework is None and neuronx_distributed_inference_installed:
            return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
        else:
            return None

    def init_device(self) -> None:
        self.init_distributed_environment()

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

    @property
    def do_metadata_broadcast(self) -> bool:
        return False

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    def init_distributed_environment(self):
        """Neuron uses transformers-neuronx for tensor parallelism.

        vLLM still needs the environment inited when TP/PP > 1
        """
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            1,
            1,
        )
