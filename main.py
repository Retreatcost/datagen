from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

llm = LLM(
    model="mistralai/Mistral-Nemo-Base-2407",
    max_model_len=2048,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    min_p=0.025,
    top_k=0,
    repetition_penalty=1.05,
    max_tokens=1024,
)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]


def print_outputs(outputs: list[RequestOutput], prompts: list):
    assert len(outputs) == len(prompts)
    print("\nGenerated Outputs:\n" + "-" * 80)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompts[i]!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)


outputs = llm.chat(conversation, sampling_params)

print_outputs(
    outputs,
    [
        conversation,
    ],
)
