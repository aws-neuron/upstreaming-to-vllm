# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# NOTE: This example is expected to work with Trn2 neuron instances with LNC2 support.

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
    model_checkpoint = "/home/ubuntu/models/open_llama_7b/"
    tp = 64
    batch_size = 4
    seq_len = 1024
    block_size = 32
    block_count = 2 * batch_size * seq_len // block_size
    override_neuron_config= {
        "enable_bucketing": False,
    }

    # Create an LLM with prefix caching enabled.
    prefix_cached_llm = LLM(
        model=model_checkpoint,
        speculative_config={
            "model": "/home/ubuntu/models/open_llama_3b/",
            "num_speculative_tokens": 5,
            "max_model_len": seq_len,
        },
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
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
