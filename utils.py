from collections import deque
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper
)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

model_name_or_path = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    #torch_dtype=torch.float16,
    load_in_8bit=True,
)
model.eval()

tokenized = lambda txt: tokenizer.encode( 
    txt, 
    return_tensors="pt", 
    add_special_tokens=False
).to('cuda')
untokenize = lambda ids: tokenizer.batch_decode(
    ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def soft_p_list(in_seq_ids, t, num_place_holder):
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits_list = model( in_seq_ids )["logits"][0][-num_place_holder-1:].detach()
        
    return [
        torch.softmax( logits/t, dim=0 ).to("cpu")
        for logits in logits_list
    ]
