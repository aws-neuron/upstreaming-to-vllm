# SPDX-License-Identifier: Apache-2.0
import os

from vllm import LLM, SamplingParams

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128"

# Sample prompts.
prompts = [
    "Can you analyze the environmental and economic impacts of transitioning "
    "from fossil fuels to renewable energy sources in developing countries "
    "over the next decade? Please consider factors like infrastructure costs, "
    "job creation, and climate change mitigation.",
    "Imagine you're designing a futuristic transportation system for a city of "
    "5 million people in the year 2050. Describe the key technologies, "
    "infrastructure requirements, and potential challenges in implementing "
    "this system while considering environmental sustainability, accessibility,"
    " and social equity factors.",
]
# Create a sampling params object.
sampling_params = SamplingParams(top_k=1, max_tokens=128)

# Create an LLM.
llm = LLM(
    # TODO: Model name unsupported with neuronx-distributed framework.
    model="/home/ubuntu/model_hf/llama-3.1-8b/",
    max_num_seqs=2,
    # To enable block KV layout in neuron, set the `is_block_kv_layout` to
    # `True` in override_neuron_config. Otherwise, the `block_size` will be
    # overridden to be the same as the max_mode_len.
    max_model_len=128,
    block_size=32,
    override_neuron_config={
        "is_prefix_caching": True,
        "is_block_kv_layout": True,
        "enable_bucketing": False,
        "on_device_sampling_config": {
            "deterministic": False,
            "do_sample": True,
            "dynamic": True,
            "global_topk": 1,
            "on_device_sampling_config": True,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0
        }
    },
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    use_v2_block_manager=True,
    tensor_parallel_size=32)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
