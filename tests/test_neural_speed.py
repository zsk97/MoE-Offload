from transformers import AutoTokenizer, TextStreamer
from transformers import AutoModelForCausalLM as HF_AutoModelForCausalLM
import time
import torch
# from intel_extension_for_transformers.transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
# model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
model_name = "facebook/opt-125m"
prompt = "input tensor in the given dimension dim"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
# input_ids = torch.randint(0, 100, (32, 64)).long()
streamer = TextStreamer(tokenizer)

print("Inputs length ", inputs.shape)


# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", load_in_4bit=True)
model = HF_AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
for i in range(3):
    outputs = model(inputs)

start = time.time()
# outputs = model.generate(inputs, streamer=streamer, max_new_tokens=30)
outputs = model(inputs)
duration = time.time() - start
print("Duration ", duration)


# intel_logits = torch.Tensor(outputs) # bs,
intel_logits = torch.Tensor(outputs.logits)[:, -1, :]
print(torch.topk(intel_logits, 10, dim=-1))

model_HF = HF_AutoModelForCausalLM.from_pretrained(model_name)

start = time.time()
outputs_HF = model_HF(inputs)
duration = time.time() - start
print("Duration ", duration)

HF_logits = torch.Tensor(outputs_HF.logits)[:, -1, :] # bs,
print("Logi shape ", HF_logits.shape)
print(torch.topk(HF_logits, 10, dim=-1))
# start = time.time()
# input_ids = torch.randint(0, 100, (32, 1)).long()
# outputs = model(input_ids=input_ids, past_key_values=outputs.past_key_values)
# duration = time.time() - start
# print("Duration ", duration)