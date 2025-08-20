# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "/home/ubuntu/models/Llama-3.1-8B-Instruct/"
# LoRA checkpoint paths.
LORA_PATH_1 = "/home/ubuntu/models/abliterated-lora/"
LORA_PATH_2 = "/home/ubuntu/models/topic-control-lora/"

# Sample prompts.
prompts = ["Hello, my name is", "The president of the United States is"]

# Create a sampling params object.
sampling_params = SamplingParams(top_k=1)

# Create an LLM with multi-LoRA serving.
llm = LLM(
    model=MODEL_PATH,
    max_num_seqs=2,
    max_model_len=64,
    tensor_parallel_size=32,
    device="neuron",
    override_neuron_config={
        "sequence_parallel_enabled": False,
        "lora_modules": {"lora_id_1": LORA_PATH_1, "lora_id_2": LORA_PATH_2},
    },
    enable_lora=True,
    max_loras=2,
)
"""
NxD Inference enables static loading of LoRA adapters: https://docs.vllm.ai/en/v0.9.0/features/lora.html on vLLM server start and does
not optionally support dynamic serving of LoRA adapters: https://docs.vllm.ai/en/v0.9.0/features/lora.html#dynamically-serving-lora-adapters
Only the lora_name needs to be specified.
The lora_id and lora_path are supplied at the LLM class/server initialization, after which the paths are
handled by NxD Inference.
"""
lora_req_1 = LoRARequest("lora_id_1", 1, LORA_PATH_1)
lora_req_2 = LoRARequest("lora_id_2", 2, LORA_PATH_2)
outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1, lora_req_2])

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
