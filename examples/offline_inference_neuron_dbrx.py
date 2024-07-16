from vllm import LLM, SamplingParams


if __name__ == "__main__":
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1)

    # Create an LLM.
    llm = LLM(
        model="databricks/dbrx-instruct",
        tensor_parallel_size=32,
        max_num_seqs=4,
        # The max_model_len and block_size arguments are required to be same as max sequence length,
        # when targeting neuron device. Currently, this is a known limitation in continuous batching
        # support in transformers-neuronx.
        # TODO(liangfu): Support paged-attention in transformers-neuronx.
        max_model_len=64,
        block_size=64,
        dtype="bfloat16",
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection, or explicitly assigned.
        device="neuron")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
