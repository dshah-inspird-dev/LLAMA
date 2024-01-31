from transformers import AutoTokenizer
import transformers 
import torch
model = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print('srat')
prompt = "Write a technical essay about Airplane"
formatted_prompt = (
    f"### Human: {prompt}### Assistant:"
)


sequences = pipeline(
    formatted_prompt,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
    num_beams=1,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
