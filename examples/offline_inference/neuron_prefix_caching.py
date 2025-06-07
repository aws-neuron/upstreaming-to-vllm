# SPDX-License-Identifier: Apache-2.0
"""
This example is expected to work with Trn2 neuron instances with LNC2 support.
It first runs all the prompt with neuron without enabling prefix caching. This
would not enable block KV cache within Neuron. Next if again runs the same
prompts with common prefix with prefic caching enabled. This time block KV
cache is enabled in neuron to support prefix caching. At the end, both the
prior generated text with prefix caching disabled is compared against
generated text when prefix caching is enabled.
"""
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


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
    "the following paragraph: "
)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

generating_prompts = [prefix + prompt for prompt in prompts]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)


def main():
    model_checkpoint = "/home/ubuntu/models/llama-3.3-70b-instruct"
    tp = 64
    batch_size = 4
    seq_len = 1024
    block_size = 32
    block_count = 2 * batch_size * seq_len // block_size
    override_neuron_config= {
            "mlp_kernel_enabled": True,
            "qkv_kernel_enabled": True,
            "attn_kernel_enabled": True,
            "fused_qkv": True,
            "attn_block_tkg_nki_kernel_enabled": True,
            "attn_block_tkg_nki_kernel_cache_update": True,
            "enable_bucketing": False,
            "cc_pipeline_tiling_factor": 1,
    }
    # Create an LLM without prefix caching as a baseline.
    # This baseline does not use block kv cache layout.
    regular_llm = LLM(
        model=model_checkpoint,
        tensor_parallel_size=tp,
        max_num_seqs=batch_size,
        max_model_len=seq_len,
        block_size=seq_len,
        device="neuron",
        override_neuron_config=override_neuron_config,
    )

    print("Results without `enable_prefix_caching`")

    # ruff: noqa: E501
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = regular_llm.generate(generating_prompts, sampling_params)

    regular_generated_texts = []
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        regular_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Destroy the LLM object and free up the GPU memory.
    del regular_llm
    cleanup_dist_env_and_memory()

    # Create an LLM with prefix caching enabled.
    prefix_cached_llm = LLM(
        model=model_checkpoint,
        tensor_parallel_size=tp,
        max_num_seqs=batch_size,
        max_model_len=seq_len,
        block_size=block_size,
        device="neuron",
        enable_prefix_caching=True,
        override_neuron_config=override_neuron_config,
        num_gpu_blocks_override=block_count,
    )

    # Warmup so that the shared prompt's KV cache is computed.
    prefix_cached_llm.generate(generating_prompts[0], sampling_params)

    # Generate with prefix caching.
    outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

    print("Results with `enable_prefix_caching`")

    cached_generated_texts = []
    # Print the outputs. You should see the same outputs as before.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Compare the results
    generated_same = all(
        [
            regular_generated_texts[i] == cached_generated_texts[i]
            for i in range(len(prompts))
        ]
    )
    print(f"Generated answers are the same: {generated_same}")


if __name__ == "__main__":
    main()

