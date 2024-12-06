import os
import torch
from PIL import Image
from typing import List

from vllm import LLM, SamplingParams
from vllm import TokensPrompt
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig
from neuronx_distributed_inference.models.mllama.modeling_mllama import MllamaInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.models.mllama.utils import create_vision_mask, get_image, get_image_tensors, add_instruct

MODEL_PATH = "/data/models/Llama-3.2-11B-Vision-Instruct"
DEVICE = "neuron"
TENSOR_PARALLEL_SIZE = 32
BATCH_SIZE = 4
CONTEXT_ENCODING_BUCKETS = [1024, 2048]
TOKEN_GENERATION_BUCKETS = [1024, 2048]
SEQ_LEN = 2048


def get_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img

def get_VLLM_mllama_model_inputs(tokenizer, config, prompt, single_image):
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    input_ids = tokenizer(instruct_prompt).input_ids
    inputs = TokensPrompt(prompt_token_ids=input_ids)
    inputs["multi_modal_data"] = {"image": input_image}
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1, temperature=1.0, top_p=1.0,
                                    max_tokens=300,
                                    stop_token_ids=config.eos_token_id)
    return inputs, sampling_params

def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Wrap the script to avoid RuntimeError during compilation
if __name__ == '__main__':
    # pseudo config to read vision_chunk_size and vision_max_num_chunks for image processing
    # actual config will be generated/read from trace_model_path in LlamaForCausalLM.load_weights_nxd()
    config = MllamaInferenceConfig(
        neuron_config=MultimodalVisionNeuronConfig(),
        load_config=load_pretrained_config(MODEL_PATH),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Create an LLM.
    llm = LLM(
        model=MODEL_PATH,
        max_num_seqs=BATCH_SIZE,
        max_model_len=SEQ_LEN,
        block_size=SEQ_LEN,
        device=DEVICE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        override_neuron_config={
            "context_encoding_buckets": CONTEXT_ENCODING_BUCKETS,
            "token_generation_buckets": TOKEN_GENERATION_BUCKETS,
        }
    )

    prompts = ["What is in this image? Tell me a story",
              "What is the recipe of mayonnaise in two sentences?" ,
              "How many people are in this image?",
              "What is the capital of Italy famous for?"]

    images = [get_image("nxdi_mm_data/dog.jpg"),
              torch.empty((0,0)), #torch.empty((0,0)),
              get_image("nxdi_mm_data/people.jpg"),
              torch.empty((0,0)), #torch.empty((0,0))
             ]

    batched_inputs = []
    for pmpt, img in zip(prompts, images):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(tokenizer, config, pmpt, img)
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)

    # test batch-size = 4
    outputs = llm.generate(batched_inputs, sampling_params)
    print_outputs(outputs)