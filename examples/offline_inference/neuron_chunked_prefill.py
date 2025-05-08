# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    ("It is not the critic who counts; not the man who points out how the "
     "strong man stumbles, or where the doer of deeds could have done them "
     "better. The credit belongs to the man who is actually in the arena, "
     "whose face is marred by dust and sweat and blood; who strives "
     "valiantly; who errs, who comes short again and again, because there "
     "is no effort without error and shortcoming; but who does actually "
     "strive to do the deeds; who knows great enthusiasms, the great "
     "devotions; who spends himself in a worthy cause; who at the best "
     "knows"),
    ("Do not go gentle into that good night, Old age should burn and rave "
     "at close of day; Rage, rage against the dying of the light. Though "
     "wise men at their end know dark is right, Because their words had "
     "forked no lightning they Do not go gentle into that good night. Good "
     "men, the last wave by, crying how bright Their frail deeds might have "
     "danced in a green bay, Rage, rage against the dying of the light. "
     "Wild men who caught and sang the sun in flight, And learn, too late, "
     "they grieved it on its way, Do not go gentle into that good night. "
     "Grave men, near death, who see with blinding sight Blind eyes could "
     "blaze like meteors and be gay, Rage, rage against the dying of the "
     "light. And you, my father, there on the sad height, Curse, bless, me "
     "now with your fierce tears, I pray. Do not go gentle into that good "
     "night. Rage, rage against the dying of the light."),
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=30, top_k=1)

# Create an LLM.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    max_model_len=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=256,  # chunk size
    block_size=32,
    tensor_parallel_size=32,

    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    override_neuron_config={
        "on_device_sampling_config": None,
    },
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
