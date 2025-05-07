# SPDX-License-Identifier: Apache-2.0
from time import time

from vllm import LLM, SamplingParams

# NOTE: This is just a running example. For benchmarking purpose,
# please see benchmarks/benchmark_prefix_caching.py

# Common prefix.
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

generating_prompts = [prefix + prompt for prompt in prompts]

# Create a sampling params object.
sampling_params = SamplingParams(top_k=1, temperature=1.0, max_tokens=256)

# Create an LLM without prefix caching as a baseline.
regular_llm = LLM(
    # TODO: Model name unsupported with neuronx-distributed framework.
    model="/home/ubuntu/model_hf/llama-3.1-8b/",
    max_num_seqs=2,
    # To enable block KV layout in neuron, set the `is_block_kv_layout` to
    # `True` in override_neuron_config. Otherwise, the `block_size` will be
    # overridden to be the same as the max_mode_len.
    max_model_len=256,
    block_size=256,
    override_neuron_config={"enable_bucketing": False},
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=32)

print("Results without `enable_prefix_caching`")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time_regular = time()
outputs = regular_llm.generate(generating_prompts, sampling_params)
duration_regular = time() - start_time_regular

regular_generated_texts = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    regular_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# # Destroy the LLM object and free up the GPU memory.
del regular_llm

# Create an LLM with prefix caching enabled.
prefix_cached_llm = llm = LLM(
    # TODO: Model name unsupported with neuronx-distributed framework.
    model="/home/ubuntu/model_hf/llama-3.1-8b/",
    # Enable prefix caching
    enable_prefix_caching=True,
    max_num_seqs=2,
    # To enable block KV layout in neuron, set the `is_block_kv_layout` to
    # `True` in override_neuron_config. Otherwise, the `block_size` will be
    # overridden to be the same as the max_mode_len.
    max_model_len=256,
    block_size=32,
    override_neuron_config={"enable_bucketing": False},
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=32)

# Warmup so that the shared prompt's KV cache is computed.
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# Generate with prefix caching.
start_time_cached = time()
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
duration_cached = time() - start_time_cached

print("Results with `enable_prefix_caching`")

cached_generated_texts = []
# Print the outputs. You should see the same outputs as before.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    cached_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Compare the results and display the speedup
generated_same = all([
    regular_generated_texts[i] == cached_generated_texts[i]
    for i in range(len(prompts))
])
print(f"Generated answers are the same: {generated_same}")

speedup = round(duration_regular / duration_cached, 2)
print(f"Speed up of cached generation compared to the regular is: {speedup}")
