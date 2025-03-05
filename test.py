from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "google/gemma-7b"  # Replace with your specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config, trust_remote_code=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_text = "Answer the following question: What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

try:
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    print(f"Error: {e}")

print("Test complete.")
