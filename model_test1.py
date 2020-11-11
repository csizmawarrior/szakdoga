import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

#we add 2 more to it, because we will use it to measure the sentences for
#the tokenizer, which has a starting and an end character additionally

def longest_sentence_size(sentences):
    current_max=0
    for sentence in sentences:
        if(len(sentence)>current_max):
            current_max=len(sentence)
    return current_max+2

#it wants the sentences as a list already split to words/punctuation
#in the first column
def text_tokenizing(text_dataframe):
    input_ids=[]
    attention_masks=[]
    targets=[]
    for_preparing=tokenizer("")
    start_symbol=for_preparing.input_ids[0]
    end_symbol=for_preparing.input_ids[-1]
    
    for i in range(text_dataframe["sentence"].size):
        target_idx_counter=text_dataframe["target_idx"][i]
        input_ids_sentence=[]
        attention_masks_sentence=[]
        input_ids_sentence.append(start_symbol)
        attention_masks_sentence.append(1)
        
        for j in range(len(text_dataframe["sentence"][i])):
            
            #a word is text_daraframe.iloc[i,0][j]
            tokenized_word=tokenizer(
                                     text_dataframe["sentence"][i][j],
                                     add_special_tokens=False
                                    )
            
            input_id=tokenized_word.input_ids
            attention_mask=tokenized_word.attention_mask
                
            if(text_dataframe["sentence"][i][j]==text_dataframe["word"][i]):
                target=len(input_id)
                text_dataframe["target_idx"][i]=len(input_ids_sentence)           
            
            input_ids_sentence.extend(input_id)
            attention_masks_sentence.extend(attention_mask)
        
        input_ids_sentence.append(end_symbol)
        attention_masks_sentence.append(1)
        input_ids.append(input_ids_sentence)
        attention_masks.append(attention_masks_sentence)
        targets.append(target)
        
    #padding    
    max_length_ids=longest_sentence_size(input_ids)
    for i in range(len(input_ids)):
        current_size=len(input_ids[i])
        additional_zeros=[]
        for j in range(max_length_ids-current_size):
            additional_zeros.append(0)
        
        input_ids[i]=input_ids[i][0:-1]+additional_zeros+[input_ids[i][-1]]
        attention_masks[i]=attention_masks[i][0:-1]
                +additional_zeros
                +[attention_masks[i][-1]]

    tokenized_text=pd.DataFrame({
        'sentence': text_dataframe["sentence"],
        'word': text_dataframe["word"],
        'target_idx': text_dataframe["target_idx"],
        'target_length': targets,
        'type': text_dataframe["type"],
        'input_id': input_ids,
        'attention_mask': attention_masks
    })
    return tokenized_text

#data reading in the original model
#train data
train_data = pd.read_csv(
    "train.tsv",
    sep='\t',
    names=['sentence',
           'word',
           'target_idx',
           'type'
          ],
    quoting=3
)

train_sentences = train_data.sentence.str.split(" ").to_frame()

train_data["sentence"]=train_sentences
train_data

#dev data

dev_data = pd.read_csv(
    "dev.tsv",
    sep='\t',
    names=['sentence',
           'word',
           'target_idx',
           'type'
          ]
)
dev_sentences = dev_data.sentence.str.split(" ").to_frame()
dev_data["sentence"] = dev_sentences
dev_data

#Letting data through the model
#and tokenizer, in the original model

#dev data

tokenized_dev_data=text_tokenizing(dev_data)
dev_outputs=[]

#every sentence gets turned into tensor
#so we can let it through the mode, then save the output
for i in range(tokenized_dev_data['input_id'].size):

    sentence=[tokenized_dev_data["input_id"][i],
              tokenized_dev_data["attention_mask"][i]]
    tensor_sentence=torch.LongTensor(sentence)

    with torch.no_grad():
        output=model(
            tensor_sentence,
            output_hidden_states=True
        )

    dev_outputs.append(output)

#train data

tokenized_train_data=text_tokenizing(train_data)
train_outputs=[]

for i in range(tokenized_train_data['input_id'].size):

    sentence=[tokenized_train_data["input_id"][i],
              tokenized_train_data["attention_mask"][i]]

    tensor_sentence=torch.LongTensor(sentence)
    print(tensor_sentence.size())

    with torch.no_grad():
        output=model(
            tensor_sentence,
            output_hidden_states=True,
        )

    train_outputs.append(output)

train_outputs

#getting out the searched tokens from the output
#in the original model,

dev_first_half = pd.DataFrame({})
for i in range(len(dev_outputs)):
    dev_first_half.insert(i, i, dev_outputs[i][2][0][0][(tokenized_dev_data["target_idx"][i]
        + tokenized_dev_data["target_length"][i]
        - 1)], True)

dev_first_half = pd.DataFrame({})
for i in range(len(dev_outputs)):
    dev_first_half.insert(i, i, dev_outputs[i][2][0][0][(tokenized_dev_data["target_idx"][i]
        + tokenized_dev_data["target_length"][i]
        - 1)], True)

#turning them into tensors
#for the neural network used in probing

dev_x = torch.tensor(dev_first_half.values)
train_x = torch.tensor(train_first_half.values)
