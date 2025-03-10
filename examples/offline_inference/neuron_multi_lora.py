# SPDX-License-Identifier: Apache-2.0

import os
import traceback

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.lora.request import LoRARequest

LORA_CKPT_PATH = "~/models/llama-3.1-8b-lora-adapter"
# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_RT_DBG_RDH_CC'] = '0'
os.environ['NEURON_RT_INSPECT_ENABLE'] = '0'
os.environ['XLA_HANDLE_SPECIAL_SCALAR'] = '1'
os.environ['UNSAFE_FP8FNCAST'] = '1'


def run_vllm(prompts, model_name_or_path, n_positions, max_batch_size,
             tp_degree):
    sampling_params = SamplingParams(top_k=1)
    override_neuron_config = {
        "sequence_parallel_enabled": False,
    }
    llm = LLM(model=model_name_or_path,
              tensor_parallel_size=tp_degree,
              max_num_seqs=max_batch_size,
              max_model_len=n_positions,
              use_v2_block_manager=True,
              override_neuron_config=override_neuron_config,
              lora_modules=[
                  LoRAModulePath(name="lora_id_1", path=LORA_CKPT_PATH),
                  LoRAModulePath(name="lora_id_2", path=LORA_CKPT_PATH)
              ],
              enable_lora=True,
              max_loras=2,
              max_lora_rank=256,
              device="neuron")
    """For multi-lora requests using NxDI as the backend, only the lora_name 
    needs to be specified. The lora_id and lora_path are supplied at the LLM 
    class/server initialization, after which the paths are handled by NxDI"""
    lora_req_1 = LoRARequest("lora_id_1", 0, " ")
    lora_req_2 = LoRARequest("lora_id_2", 1, " ")
    # outputs = llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts,
                           sampling_params,
                           lora_request=[lora_req_1, lora_req_2])
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return [output.prompt + output.outputs[0].text for output in outputs]


def vllm_integ_test(model,
                    n_positions,
                    max_batch_size,
                    tp_degree,
                    model_name_or_path=None):
    if model_name_or_path is None:
        raise ValueError(
            "Please provide either 's3_model_dir' or 'model_name_or_path' in "
            "the test definition")

    # Sample prompts.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
    ]
    vllm_outputs = run_vllm(prompts, model_name_or_path, n_positions,
                            max_batch_size, tp_degree)

    # check if actual outputs is the prefix of expected output
    # we cannot compare the full sequence as each actual outputs has different
    # length due to continuous batching
    for actual_seq in vllm_outputs:
        print(f"actual: {actual_seq}")


def run_vllm_integration_test(param, **kwargs):
    model = param["model"]
    model_path = param["model_path"]

    results = {}
    try:
        vllm_integ_test(
            model,
            param["max_context_length"] + param["max_new_tokens"],
            param["batch_size"],
            param["tp_degree"],
            model_name_or_path=model_path,
        )
        results["inference_success"] = True
    except Exception:
        print(traceback.format_exc())
        results["inference_success"] = False

    return results


def main():
    param = {
        "max_context_length": 256,
        "max_new_tokens": 256,
        "batch_size": 4,
        "tp_degree": 32,
        "model_path": "/home/ubuntu/models/llama-3.1-8b",
        "model": "llama-3.1-8b",
    }
    run_vllm_integration_test(param)


if __name__ == "__main__":
    main()
