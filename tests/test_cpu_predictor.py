import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import MoEModelOutput

model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
input_ids = torch.randint(0, 100, (32, 64)).long()
attention_mask = torch.ones((32, 64)).long()
decode_input_ids = torch.zeros((32, 1)).long()

torch.set_num_threads(16)
# Check PyTorch threads
print(f"PyTorch threads: {torch.get_num_threads()}")

# os.environ['MKL_NUM_THREADS'] = '16'
# os.environ['OPENBLAS_NUM_THREADS'] = '16'
# os.environ['NUMEXPR_NUM_THREADS'] = '16'

# Check environment variables for underlying libraries
print(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS')}")
print(f"OPENBLAS_NUM_THREADS: {os.getenv('OPENBLAS_NUM_THREADS')}")
print(f"NUMEXPR_NUM_THREADS: {os.getenv('NUMEXPR_NUM_THREADS')}")

# print(f"Default number of CPU threads: {default_threads}")
for i in range(3):
    outputs = model(input_ids, attention_mask, decode_input_ids)

start = time.time()
outputs = model(input_ids, attention_mask, decode_input_ids)
duration = time.time() - start

print("Prefill Time ", duration)

start = time.time()
for i in range(100):
    # start = time.time()
    encoder_outputs = MoEModelOutput(last_hidden_state=outputs.encoder_last_hidden_state,
                                    hidden_states=outputs.encoder_hidden_states,
                                    attentions=outputs.encoder_attentions)
    outputs = model(decoder_input_ids=decode_input_ids, encoder_outputs=encoder_outputs, past_key_values=outputs.past_key_values)
duration = time.time() - start
print("Time ", duration)