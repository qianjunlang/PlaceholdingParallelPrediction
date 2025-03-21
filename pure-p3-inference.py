import time
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm , trange
import torch

from prompt_template import get_ans , get_input
from utils import (
    model,
    tokenized, untokenize, tokenizer,
    soft_p_list,
    greedy_gen, random_gen,
    model_name_or_path,
)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def p3_pred(datasetName , sample):
    torch.cuda.empty_cache()
    with torch.no_grad():
        pred = {}
        
        for prompt_type,prompt_list in get_input( 
            dataset_name=datasetName , sample=sample 
        ).items():
            if prompt_type=="question_prompt": continue

            pred[prompt_type] = []
            for prompt in prompt_list:
                p_list = torch.softmax( 
                    model(  
                        tokenized( prompt + place_holder*num_place_holders ) 
                    )["logits"][0][-num_place_holders-1:].detach()
                , dim=1 ).cpu()#[:,label_token_ids]

                p_list = p_list[:,label_token_ids_list].unsqueeze(-1)

                pred[prompt_type].append(
                    mapping[ torch.argmax( p_list, dim=1) ].tolist()
                )

    return pred
    
#*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*main*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
dataset_names = ["ag_news"]#["yahoo_answers","dbpedia"] #"ag_news","aclImdb",
# dataset_names = ["SST2","dbpedia"]

max_num_samples = 1#1024
num_place_holders = 512
place_holder = "<unk>"


print(model_name_or_path,dataset_names)


for datasetName in dataset_names:
    print(datasetName)
    
    df = pd.read_csv("./data/"+ datasetName +"/test.csv", header=None)
    dataset_config = {
        "num_samples" : min( len(df) , max_num_samples ) ,
        "num_place_holders" : num_place_holders ,
        "prompt_format" : get_input(dataset_name=datasetName , sample=["<???>"]*5 ), #sample=[0]+df.iloc[0].tolist()
    }

    with open("./data/"+ datasetName +"/verbalizer.txt",'r',encoding="utf-8") as f:
        verbalizer = json.loads(f.readline())

        label_token_ids = {}
        
        for cls in verbalizer.keys():
            label_token_ids[cls] = tokenized(verbalizer[cls])[0].cpu()


    with open( "./result/"+model_name_or_path[11:]+"_"+datasetName+"_config&ans.jsonlines" ,"w", encoding='utf-8' ) as f:
        json.dump( dataset_config , f )
        f.write('\n')
        json.dump([
            get_ans( dataset_name=datasetName , sample=sample )
            for sample in df.itertuples()
        ] ,f)
        f.write('\n')
        
#-=-=-=-=-=-=-=-=-

    label_token_ids_list = [ int(token_id[0]) for token_id in label_token_ids.values() ]
    mapping = np.array(list(label_token_ids.keys()))

#-=-=-=-=-=-=-=-=-

    with open( "./result/"+model_name_or_path[11:]+"_"+datasetName+"_pred.jsonlines" ,"w", encoding='utf-8' ) as f:
        with tqdm(total=dataset_config["num_samples"]) as p_bar:

            for sample in df.itertuples():

                pred = p3_pred( datasetName , sample ) 

                json.dump(pred,f)
                f.write("\n")

                p_bar.update(1)
                if p_bar.n >= dataset_config["num_samples"] : break
                if 0 == p_bar.n & 255: f.flush()

                # start_time = time.time()
                # duration = time.time() - start_time
                # print(duration)
                # print(( time.time() - start_time ) - duration )
                # duration = time.time() - start_time