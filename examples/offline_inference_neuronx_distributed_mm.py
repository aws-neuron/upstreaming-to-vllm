import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm import TextPrompt

from neuronx_distributed_inference.models.mllama.utils import get_image, add_instruct

# Configurations
MODEL_PATH = "/home/ubuntu/model_hf/Llama-3.2-11B-Vision-Instruct-hf"
BATCH_SIZE = 4
TENSOR_PARALLEL_SIZE = 32
SEQ_LEN = 2048
CONTEXT_ENCODING_BUCKETS = [1024, 2048]
TOKEN_GENERATION_BUCKETS = [1024, 2048]
SEQUENCE_PARALLEL_ENABLED = False
# Model Inputs
PROMPTS = ["What is in this image? Tell me a story",
              "What is the recipe of mayonnaise in two sentences?" ,
              "How many people are in this image?",
              "What is the capital of Italy famous for?"]

IMAGES = [get_image("nxdi_mm_data/dog.jpg"),
            torch.empty((0,0)),
            get_image("nxdi_mm_data/people.jpg"),
            torch.empty((0,0)),
            ]


def get_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img

def get_VLLM_mllama_model_inputs(prompt, single_image):
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)
    inputs["multi_modal_data"] = {"image": input_image}
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1, temperature=1.0, top_p=1.0,
                                    max_tokens=300)
    return inputs, sampling_params

def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    assert len(PROMPTS) == len(IMAGES), \
        f"Text and image prompts should have the same batch size, got {len(PROMPTS)} and {len(IMAGES)}"

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
        }
    )

    batched_inputs = []
    for pmpt, img in zip(PROMPTS, IMAGES):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(pmpt, img)
        # test batch-size = 1
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)

    # test batch-size = 4
    outputs = llm.generate(batched_inputs, sampling_params)
    print_outputs(outputs)