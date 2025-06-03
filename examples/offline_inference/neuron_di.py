# SPDX-License-Identifier: Apache-2.0

# This example can be run by having two terminals
# For the prefill terminal:
# SEND=1 python neuron_di.py
# For the decode terminal:
# python neuron_di.py

# This example assumes you have the llama-3.2-3b model
# downloaded into a folder called "models" under your home directory

import logging
import os
import subprocess
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

logging.basicConfig(level=logging.DEBUG)

MODEL_NAME = "llama-3.2-3b"
CURRENT_DIR = Path.cwd()
LOCAL_MODEL = CURRENT_DIR / "models" / MODEL_NAME
COMPILED_MODEL = CURRENT_DIR / "compiled_models/llama32di_compiled"

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT'] = '45645'
os.environ['NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED'] = '1'


def sender_routine():
    # use first 16 cores for prefill
    os.environ['NEURON_RT_VISIBLE_CORES'] = '0-15'
    kv_transfer_config = KVTransferConfig(
        kv_connector='NeuronConnector',
        kv_buffer_device='cpu',
        kv_buffer_size=1e9,
        kv_rank=0,
        kv_role="kv_producer",
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=8199,
        neuron_core_offset=0,
    )
    results = common_routine(kv_transfer_config)
    return results


def recver_routine():
    # use next 16 cores for decode
    os.environ['NEURON_RT_VISIBLE_CORES'] = '16-31'
    kv_transfer_config = KVTransferConfig(
        kv_connector='NeuronConnector',
        kv_buffer_device='cpu',
        kv_buffer_size=1e9,
        kv_rank=1,
        kv_role="kv_consumer",
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=8199,
        neuron_core_offset=16,
    )
    results = common_routine(kv_transfer_config)
    return results


def common_routine(kv_transfer_config):
    # Sample prompts.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Answer in five words: First five planets from sun are",
        "sushi tastes good with",
        "Ethiopia is a country in",
        "two plus two is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1,
                                     temperature=1.0,
                                     top_p=1.0,
                                     max_tokens=20)

    # Create an LLM. Update config as per the case
    llm = LLM(
        model=str(LOCAL_MODEL),
        max_num_seqs=2,
        max_model_len=256,
        # The device can be automatically detected when AWS Neuron SDK
        # is installed. The device argument can be either unspecified
        # for automated detection, or explicitly assigned.
        device="neuron",
        tensor_parallel_size=16,
        kv_transfer_config=kv_transfer_config,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    results = []
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return results


def compile():
    if not COMPILED_MODEL.exists():
        result = subprocess.run(
            Path(__file__).parent / "compile_di_single_instance.sh")
        assert result.returncode == 0, "compile script failed"


def run_neuron_di_example():
    os.environ["NEURON_COMPILED_ARTIFACTS"] = str(COMPILED_MODEL)
    if os.getenv("SEND", None) == '1':
        compile()
        sender_routine()
    else:
        recver_routine()


if __name__ == "__main__":
    run_neuron_di_example()
