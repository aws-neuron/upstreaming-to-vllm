from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.neuronx_distributed import get_neuron_model
from vllm.worker.neuron_model_runner import NeuronModelRunner

logger = init_logger(__name__)


class NeuronxDistributedModelRunner(NeuronModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ):
        super().__init__(model_config, parallel_config, scheduler_config, device_config)

    def load_model(self) -> None:
        self.model = get_neuron_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config)
