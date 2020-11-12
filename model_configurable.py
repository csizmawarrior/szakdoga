
import hydra
import numpy as np
from torchvision import datasets
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="model_config")
def model_configurable(cfg: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    model = AutoTokenizer.from_pretrained(cfg.model.model_name)

#train data
    train_data = pd.read_csv(
    cfg.data.path+cfg.data.train,
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

#dev data
    dev_data = pd.read_csv(
        cfg.data.path+cfg.data.dev,
        sep='\t',
        names=['sentence',
           'word',
           'target_idx',
           'type'
          ]
    )
    dev_sentences = dev_data.sentence.str.split(" ").to_frame()
    dev_data["sentence"] = dev_sentences
    print(dev_data)


if __name__ == "__main__":
    my_app()


