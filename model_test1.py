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


#getting out the searched tokens from the output
#in the original model,

dev_first_half = pd.DataFrame({})
for i in range(len(dev_outputs)):
    dev_first_half.insert(i, i, dev_outputs[i][2][0][0][(tokenized_dev_data["target_idx"][i]
        + tokenized_dev_data["target_length"][i]
        - 1)], True)

train_first_half = pd.DataFrame({})
for i in range(len(train_outputs)):
    train_first_half.insert(i,i,train_outputs[i][2][0][0][(
        tokenized_train_data["target_idx"][i]
        + tokenized_train_data["target_length"][i]
        - 1)], True)

#turning them into tensors
#for the neural network used in probing

dev_x = torch.tensor(dev_first_half.values)
train_x = torch.tensor(train_first_half.values)

#labels from the dev data
labels = tokenized_dev_data["type"].unique()


#the expected output from the prober
#dev data

text_dev_y = tokenized_dev_data["type"]
text_dev_y

dev_y=[]
for i in range(text_dev_y.size):
    dev_y.append(np.where(labels == text_dev_y[i])[0][0])

dev_y=torch.tensor(dev_y)

#train data

text_train_y = tokenized_train_data["type"]
text_train_y

train_y=[]
for i in range(text_train_y.size):
    train_y.append(np.where(labels == text_train_y[i])[0][0])

train_y=torch.tensor(train_y)

#the batch iterator class


class BatchIterator:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def iterate_once(self):
        for start in range(0, len(self.x), self.batch_size):
            end = start + self.batch_size
            yield self.x[start:end], self.y[start:end]

#the instance of the batch for the probing

rain_iter = BatchIterator(train_x, train_y, 200)

#the classifier/model class


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

#the model itself


model = Classifier(
    input_dim = train_x.size(1),
    output_dim = labels.size,
    hidden_dim = 50
)


#the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#the training loop itself, printing out the 10 epochs
#the amount it learns now

batch_size = 50
train_iter = BatchIterator(train_x, train_y, batch_size)
dev_iter = BatchIterator(dev_x, dev_y, batch_size)

all_train_loss = []
all_dev_loss = []
all_train_acc = []
all_dev_acc =[]

n_epochs = 10
for epoch in range(n_epochs):
    #training loop
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
