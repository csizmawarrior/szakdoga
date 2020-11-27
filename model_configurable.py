
import math
import hydra
import numpy as np
from torchvision import datasets
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

#we add 2 more to it, because we will use it to measure the sentences for
#the tokenizer, which has a starting and an end character additionally
def longest_sentence_size(sentences):
    current_max=0
    for sentence in sentences:
        if(len(sentence)>current_max):
            current_max=len(sentence)
    return current_max+2

def text_tokenizing(text_dataframe, tokenizer):
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
        attention_masks[i]=attention_masks[i][0:-1]+additional_zeros+[attention_masks[i][-1]]
            
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

class BatchIterator_model:
    def __init__(self, id, ma, ti, tl, batch_size):
        self.id = id
        self.ma = ma
        self.ti = ti
        self.tl = tl
        self.batch_size = batch_size

    def iterate_once(self):
        for start in range(0, len(self.id), self.batch_size):
            end = start + self.batch_size
            if(end > len(self.id)):
                end= len(self.id)
            yield   self.id[start:end], self.ma[start:end], self.ti[start:end], self.tl[start:end]



@hydra.main(config_path="conf")
def model_configurable(cfg: DictConfig) -> None:
    print(cfg.data.train)
    print(cfg.data.test)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    model = AutoModel.from_pretrained(cfg.model.model_name)

#train data
    train_path = cfg.data.path+"/"+cfg.data.train

    train_data = pd.read_csv(
        train_path,
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


    tokenized_train_data=text_tokenizing(train_data, tokenizer)

    train_batch_size = math.ceil(tokenized_train_data["input_id"].size/50)
    train_outputs=pd.DataFrame({})

#we get out every target token's last token by batching
#the input
    model_train_iterator = BatchIterator_model(
                            torch.tensor(tokenized_train_data["input_id"]),
                            torch.tensor(tokenized_train_data["attention_mask"]),
                            torch.tensor(tokenized_train_data["target_idx"]),
                            torch.tensor(tokenized_train_data["target_length"]),
                            train_batch_size
                            )


    for batch_id, batch_ma, batch_ti, batch_tl in model_train_iterator.iterate_once():

        batch_id_tensor=torch.LongTensor(batch_id)
        batch_ma_tensor=torch.LongTensor(batch_ma)

        with torch.no_grad():
            output=model(
                input_ids=batch_id_tensor,
                attention_mask=batch_ma_tensor,
                output_hidden_states=True,
                return_dict=True
            )

        for i in range(len(batch_id_tensor)):

            target_index = (batch_ti[i] + batch_tl[i] - 1)
            train_outputs.insert(
               len(train_outputs.columns),
               len(train_outputs.columns),
               output[2][0][i][target_index],
               True)

    train_x=torch.tensor(train_outputs.values)
    train_x=torch.transpose(train_x, 0,1)

#dev data
    dev_data = pd.read_csv(
        cfg.data.path+"/"+cfg.data.dev,
        sep='\t',
        names=['sentence',
               'word',
               'target_idx',
               'type'
              ],
        quoting=3,
    )
    dev_sentences = dev_data.sentence.str.split(" ").to_frame()
    dev_data["sentence"] = dev_sentences


    tokenized_dev_data=text_tokenizing(dev_data, tokenizer)

    dev_batch_size = math.ceil(tokenized_dev_data["input_id"].size/5)
    dev_outputs=pd.DataFrame({})

#every sentence gets turned into tensor
#so we can let it through the mode, then save the output
    model_dev_iterator = BatchIterator_model(
                            torch.tensor(tokenized_dev_data["input_id"]),
                            torch.tensor(tokenized_dev_data["attention_mask"]),
                            torch.tensor(tokenized_dev_data["target_idx"]),
                            torch.tensor(tokenized_dev_data["target_length"]),
                            dev_batch_size
                            )


    for batch_id, batch_ma, batch_ti, batch_tl in model_dev_iterator.iterate_once():

        batch_id_tensor=torch.LongTensor(batch_id)
        batch_ma_tensor=torch.LongTensor(batch_ma)

        with torch.no_grad():
            output=model(
                input_ids=batch_id_tensor,
                attention_mask=batch_ma_tensor,
                output_hidden_states=True,
                return_dict=True
            )

        for i in range(len(batch_id_tensor)):

            target_index = (batch_ti[i] + batch_tl[i] - 1)

            dev_outputs.insert(
                len(dev_outputs.columns),
                len(dev_outputs.columns)-1,
                output[2][0][i][target_index],
                True
    )

    dev_x = torch.tensor(dev_outputs.values)
    dev_x = torch.transpose(dev_x, 0,1)
#preparing expected outputs (y values)

    labels = tokenized_dev_data["type"].unique()

    text_dev_y = tokenized_dev_data["type"]

    text_train_y = tokenized_train_data["type"]

    dev_y=[]
    for i in range(text_dev_y.size):
        dev_y.append(np.where(labels == text_dev_y[i])[0][0])

    train_y=[]

    for i in range(text_train_y.size):
        train_y.append(np.where(labels == text_train_y[i])[0][0])

    train_y=torch.tensor(train_y)
    dev_y=torch.tensor(dev_y)


#Batch Class
    class BatchIterator:
        def __init__(self, x, y, batch_size):
            self.x = x
            self.y = y
            self.batch_size = batch_size

        def iterate_once(self):
            for start in range(0, len(self.x), self.batch_size):
                end = start + self.batch_size
                yield self.x[start:end], self.y[start:end]

#Classifier
    class Classifier(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super().__init__()
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        def forward(self, X):
            h = self.input_layer(X)
            h = self.relu(h)
            out = self.output_layer(h)
            return out

#The explicit model
    model = Classifier(
        input_dim = train_x.size(1),
        output_dim = labels.size,
        hidden_dim = 50
    )


#criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

#First accuracy
    test_pred = model(dev_x).max(axis=1)[1]
    test_acc = torch.eq(test_pred, dev_y).sum().float() / len(dev_x)
    print(test_acc)


#training loop

    batch_size = 50
    train_iter = BatchIterator(train_x, train_y, batch_size)
    dev_iter = BatchIterator(dev_x, dev_y, batch_size)

    all_train_loss = []
    all_dev_loss = []
    all_train_acc = []
    all_dev_acc =[]

    n_epochs = 10
    for epoch in range(n_epochs):
        for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
        

            y_out = model(batch_x)
        
            loss = criterion(y_out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        train_out = model(train_x)
        train_loss = criterion(train_out, train_y)
        all_train_loss.append(train_loss.item())
        train_pred = train_out.max(axis=1)[1]
        train_acc = torch.eq(train_pred, train_y).sum().float() / len(train_x)
        all_train_acc.append(train_acc)

        dev_out = model(dev_x)
        dev_loss = criterion(dev_out, dev_y)
        all_dev_loss.append(dev_loss.item())
        dev_pred = dev_out.max(axis=1)[1]
        dev_acc = torch.eq(dev_pred, dev_y).sum().float() / len(dev_x)
        all_dev_acc.append(dev_acc)
    
        print(f"Epoch: {epoch}\n train_accuracy: {train_acc} train loss: {train_loss}")
        print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")



#def functions that are needed


if __name__ == "__main__":
    model_configurable()


