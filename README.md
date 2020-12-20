# Measuring BERT models with probing

The programs have one goal, implementing a neural network that will probe the output of the BERT model in question and return the accuracy achieved on it or the predicted types for a given data group.

It is mainly based on transformers library by huggingface and uses Python3 in the implementation. There are basic script files that help iterating through the now existing datas on any BERT model and extract the results into well distributed files.

The paper about the model is in the repository in hungarian. Named: `szakdolgozat.pdf`

## Quickstart

Requirements: hydra, Python 3, transformers and pytorch

If python3 is the only python version on the computer use `python` otherwise `python3`

To get the accuracies to the standard output:

`python +model=<model-name> +data=<data-group-name> +layer=<hidden-layer-number> `

To run on a given model every layer with every now existing data group:

`./script_try_first_half <model-name>
./script_try <model-name>`

## Extraction

To extract the programs with new BERT models or data groups, you should create new `.yaml` files.
In the case of new data groups, you have to create `dev.tsv train.tsv test.tsv` files into a new subfolder with the same name as the `.yaml` file under `conf/data` folder.

In order to use the new data groups in the scripts too, you have to write it manually in it, but i suggest to change the script to automatically use every data group in the `conf/data` folder.

The model parameter name for the scripts is the same as the name of the `.yaml` files.

## For fault measuring

To measure the failures of each models instead of accuracies, youcan extract the predicted values of the neural network for the test data, instead of its accuracy.

To do that, you should comment out the `print(test_pred)` line and comment the other `print()` commands.

The correct results are found in the given `test.tsv` files to compare it with the predicted ones.
