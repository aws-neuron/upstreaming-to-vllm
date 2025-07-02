# SPDX-License-Identifier: Apache-2.0
import requests
import torch
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from PIL import Image
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams, TextPrompt

BATCH_SIZE = 4
TEXT_SEQ_LENGTH = 10*1024
VISION_SEQ_LENGTH = 10*1024

# Model Inputs
PROMPTS = [
    "What is in this image? Tell me a story",
    "How many animals do you see? What type?",
    "Describe this image",
    "Where can I see this?",
]
IMAGES = [
    ImageAsset("blue_flowers").pil_image,
    ImageAsset("bird").pil_image,
    ImageAsset("blue_flowers").pil_image,
    ImageAsset("bird").pil_image,
]
SAMPLING_PARAMS = [
    dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256)
    for _ in range(len(PROMPTS))
]


def get_VLLM_pixtral_model_inputs(prompt, single_image, sampling_params):
    """
    Prepare all inputs for pixtral generation, including:
      1. put text prompt into instruct chat template
      2. compose single text and single image prompt into Vllm's prompt class
      3. prepare sampling parameters
    """
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image).replace("|image|", "[IMG]")
    inputs = TextPrompt(prompt=instruct_prompt)

    if input_image is not None:
        inputs["multi_modal_data"] = {"image": input_image}

    sampling_params = SamplingParams(**sampling_params)
    return inputs, sampling_params


def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt.replace("[IMG]", "")
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), \
        f"""Text, image prompts and sampling parameters should have the 
            same batch size; but got {len(PROMPTS)}, {len(IMAGES)}, 
            and {len(SAMPLING_PARAMS)}"""

    # Create an LLM.
    llm = LLM(model="/home/models/Pixtral-Large-Instruct-2411/",
              max_num_seqs=BATCH_SIZE,
              max_model_len=TEXT_SEQ_LENGTH,
              block_size=TEXT_SEQ_LENGTH,
              device="neuron",
              tensor_parallel_size=64,
              override_neuron_config={
                  "flash_decoding_enabled": True,
                  "enable_bucketing": True,
                  "context_encoding_buckets": [2*1024, 4*1024, TEXT_SEQ_LENGTH],
                  "token_generation_buckets": [2*1024, 4*1024, TEXT_SEQ_LENGTH],
                  "skip_warmup": True,
                  "torch_dtype": torch.float16,
                  "save_sharded_checkpoint": True,
                  "on_device_sampling_config": {
                      "global_topk": 1,
                      "dynamic": False,
                      "deterministic": False
                  },
                  "vision_neuron_config.tp_degree": 16,
                  "vision_neuron_config.torch_dtype": torch.float16,
                  "vision_neuron_config.buckets": [2*1024, 4*1024, 6*1024, 8*1024, VISION_SEQ_LENGTH],
              })

    batched_inputs = []
    batched_sample_params = []
    for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_VLLM_pixtral_model_inputs(
            pmpt, img, params)
        # test batch-size = 1
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)
        batched_sample_params.append(sampling_params)

    # test batch-size = 4
    outputs = llm.generate(batched_inputs, batched_sample_params)
    print_outputs(outputs)
