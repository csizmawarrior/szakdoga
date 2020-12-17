# Thesis work

## The Beginning

This repository is containing files and programs that were used to work on the project of measuring different BERT models by probing, in order to find differences, and decide which model is the better than the other, and why. The thesis was written in hungarian so some files have hungarian names. One can run tests on any BERT model that is contained in the transformers library's AutoModel and its tokenizer in the AutoTokenizer. 
The program takes a data group (that is also configurable), they contain test.tsv, train.tsv and dev.tsv files. These files contain the sentences that are the test cases for the neural networks, the target word, its position in the sentence and the class of the target word.
The program uses the BERT model to extract information from the target word, and uses its last wordpiece to feed the neural network, so it can learn from the information and guess the right class for each test case. Before training the program prints the neural network's accuracy on the dev data.
By default the program runs 10 epochs and after each epoch it prints out the accuracy of the neural network on the train data and the dev data, with their losses. It uses early stopping to prevent overfitting, after it stops, it runs the test data through the BERT and the neural network and prints out the accuracy.
For this the program is configurable with any hidden layer of the 13 hidden layers of a BERT model, and the program will use that layer to teach the neural network. 

## How to use it

You have to run the `model_configurable.py` with python3. The arguments must be given according to the hydra standard, so e.g. with the case_noun data group, the first layer, and the HuBERT BERT model the arguments will look like this: `+data=case_noun +layer=1 +model=HUBert`
The correct names of the arguments can be checked within the subfolders of the conf folder, where the `.yaml` files' names are the options for the arguments. You can add more arguments (for layers you only can, if you change the indexing in the code for all data files after using BERT model on them) by using the same structure as in the current `.yaml` files, with new `.yaml` files. For models, you can only add more option, if the given `model_name` and `tokenizer_name` are both available in the AutoModel and AutoTokenizer.
To add more data group options you are also required to follow the structure of the files, and add the new data files into the given subfolder of the data folder, or reference an existing data group (not recommended, to avoid confusion).

By this configuration the model will print the accuracies to the standard output.
You can run full tests on a BERT model with every now existing data group and the 13 layers by running the scripts `script_try` and `script_try_first_half` . These scripts will iterate through 6 and 7 of the layers with every data group now present in the `conf/data` folder. It is a manually written code, so for extension i suggest to use a more automatic one, that takes the arguments automatically from the `conf/data` folder, if you don't want to manually add the new data groups 13 times manually to the scripts.
The scripts require the model's name used in hydra configuration and will write the accuracy into the `results` folder's subfolder named after the model. The accuracies are further distributed into folders by the layer they were measured on, and each data group has an own file with the accuracies in it.
