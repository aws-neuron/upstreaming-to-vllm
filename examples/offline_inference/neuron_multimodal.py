import torch
import requests
from PIL import Image

from vllm import LLM, SamplingParams
from vllm import TextPrompt

from neuronx_distributed_inference.models.mllama.utils import add_instruct

def get_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image


# Configurations
MODEL_PATH = "/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf"
BATCH_SIZE = 4
SEQ_LEN = 2048
TENSOR_PARALLEL_SIZE = 32
CONTEXT_ENCODING_BUCKETS = [1024, 2048]
TOKEN_GENERATION_BUCKETS = [1024, 2048]
SEQUENCE_PARALLEL_ENABLED = False
IS_CONTINUOUS_BATCHING = True
ON_DEVICE_SAMPLING_CONFIG = {"global_topk":64, "dynamic": True, \
                             "deterministic": False}

# Model Inputs
PROMPTS = ["What is in this image? Tell me a story",
            "What is the recipe of mayonnaise in two sentences?" ,
            "Describe this image",
            "What is the capital of Italy famous for?",
            ]
IMAGES = [get_image("https://github.com/meta-llama/llama-models/blob/main/models/scripts/resources/dog.jpg?raw=true"),
          torch.empty((0,0)),
          get_image("https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/nxd-inference-block-diagram.jpg"),
          torch.empty((0,0)),
          ]
SAMPLING_PARAMS = [
    dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256),
    dict(top_k=1, temperature=0.9, top_p=1.0, max_tokens=256),
    dict(top_k=10, temperature=0.9, top_p=0.5, max_tokens=512),
    dict(top_k=10, temperature=0.75, top_p=0.5, max_tokens=1024),
    ]


def get_VLLM_mllama_model_inputs(prompt, single_image, sampling_params):
    # Prepare all inputs for mllama generation, including:
    # 1. put text prompt into instruct chat template
    # 2. compose single text and single image prompt into Vllm's prompt class
    # 3. prepare sampling parameters
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)
    inputs["multi_modal_data"] = {"image": input_image}
    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)
    return inputs, sampling_params

def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), \
        f"""Text, image prompts and sampling parameters should have the 
            same batch size; but got {len(PROMPTS)}, {len(IMAGES)}, 
            and {len(SAMPLING_PARAMS)}"""

    # Create an LLM.
    llm = LLM(
        model=MODEL_PATH,
        max_num_seqs=BATCH_SIZE,
        max_model_len=SEQ_LEN,
        block_size=SEQ_LEN,
        device="neuron",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        override_neuron_config={
            "context_encoding_buckets": CONTEXT_ENCODING_BUCKETS,
            "token_generation_buckets": TOKEN_GENERATION_BUCKETS,
            "sequence_parallel_enabled": SEQUENCE_PARALLEL_ENABLED,
            "is_continuous_batching": IS_CONTINUOUS_BATCHING,
            "on_device_sampling_config": ON_DEVICE_SAMPLING_CONFIG,
        }
    )

    batched_inputs = []
    batched_sample_params = []
    for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(
            pmpt, img, params
            )
        # test batch-size = 1
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)
        batched_sample_params.append(sampling_params)

    # test batch-size = 4
    outputs = llm.generate(batched_inputs, batched_sample_params)
    print_outputs(outputs)