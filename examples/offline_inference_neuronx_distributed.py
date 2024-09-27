import os

from vllm import LLM, SamplingParams

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

# Sample prompts.
prompts = [
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(top_k=1)

# Create an LLM.
llm = LLM(
    # TODO: Model name unsupported with neuronx-distributed framework.
    model="/home/ubuntu/Nxd/models/Llama-2-7b",
    max_num_seqs=4,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when targeting neuron device.
    # Currently, this is a known limitation in continuous batching support
    # in neuronx-distributed-inference.
    # TODO: Support paged-attention
    max_model_len=128,
    block_size=128,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=32)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
