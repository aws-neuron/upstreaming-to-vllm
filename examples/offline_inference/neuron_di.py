# SPDX-License-Identifier: Apache-2.0

# Set following environment variables
# NEURON_SEND_IP                        - IP of prefill node
# NEURON_RECV_IP                        - IP of receive node
# KV_IP                                 - same as NEURON_SEND_IP
# OFI_NCCL_PROTOCOL                     - set to SENDRECV
# NRT_ASYNC_SEND_RECV_BOOTSTRAP_PORT    - 45645
# NCCL_DEBUG_SUBSYS                     - ALL
# NEURON_RT_ROOT_COMM_ID                - localhost:6868
# BATCH_TRANSFER                        - 5120 (set to highest allowed value
#                                               subject to limit on number of
#                                               connections < 128 per rank)
# KV_PRODUCER                           - set to 1 on prefill node
# ASYNC_DI_PRODUCER                     - set to 1 on prefill node
# ASYNC_DI                              - set to 1 on decode node
# NEURON_COMPILED_ARTIFACTS             - set to path of pre-compiled artifacts

import logging
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

logging.basicConfig(level=logging.DEBUG)

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

# Sample prompts.
prompts = [
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Answer in five words: First five planets from sun are",
]
# Create a sampling params object.
sampling_params = SamplingParams(top_k=1,
                                 temperature=1.0,
                                 top_p=1.0,
                                 max_tokens=40)

kv_producer = os.environ.get("KV_PRODUCER", None) == "1"
kv_ip = os.environ.get("KV_IP", "127.0.0.1")

print(f"Running with kv_ip {kv_ip}, kv_producer {kv_producer}")

if kv_producer:
    # use first 16 cores for prefill
    os.environ['NEURON_RT_VISIBLE_CORES'] = '0-15'
    kv_transfer_config = KVTransferConfig(
        kv_connector='NeuronConnector',
        kv_buffer_device='cpu',
        kv_buffer_size=1e9,
        kv_rank=0,
        kv_role="kv_producer",  # this arg doesn't matter in this test
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=8199,
    )
else:
    # use next 16 cores for decode
    os.environ['NEURON_RT_VISIBLE_CORES'] = '16-31'
    kv_transfer_config = KVTransferConfig(
        kv_connector='NeuronConnector',
        kv_buffer_device='cpu',
        kv_buffer_size=1e9,
        kv_rank=1,
        kv_role="kv_consumer",  # this arg doesn't matter in this test
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=8199,
    )

# Create an LLM. Update config as per the case
llm = LLM(
    model="openlm-research/open_llama_3b",
    max_num_seqs=4,
    max_model_len=2048,
    block_size=2048,
    # The device can be automatically detected when AWS Neuron SDK is installed
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=16,
    kv_transfer_config=kv_transfer_config,
)
# Generate texts from the prompts. The output is a list of RequestOutput
# objects that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

print("generation done")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# sleep so that the TCP store will stay alive for decode program
time.sleep(100000)
