# Measuring BERT models with probing

The programs have one goal, implementing a neural network that will probe the output of the BERT model in question and return the accuracy achieved on it by the predicted types for a given data group.

It is mainly based on transformers library by huggingface and uses Python3 in the implementation. There are basic script files that help iterating through the now existing datas on every hidden layer of any BERT model and extract the results into well distributed files.

The paper about the model is in the repository. It is in hungarian. Named: `szakdolgozat.pdf`

## Quick start and running

Requirements: hydra, Python 3, transformers and pytorch

If python3 is the only python version on the computer use `python` otherwise `python3`

To get the accuracies to the standard output run:

`python +model=<model-name> +data=<data-group-name> +layer=<hidden-layer-number> +measure_type=accuracy `

To run on a given model with every layer with every now existing data group:

`./script_try_first_half <model-name>
./script_try <model-name>`

These scripts will save every accuracy results into the `results/<model-name>/<layer-name>` folder and into its given subfolder, distributed by model and layer.

## Extraction

To extract the results of the programs with new BERT models or data groups, you should create new `.yaml` files.
In the case of new data groups, you have to create `dev.tsv train.tsv test.tsv` files into a new subfolder with the same name as the `.yaml` file under `conf/data` folder.

In order to use the new data groups in the scripts too, you have to write it manually into them, but i suggest to change the script in a way when it is capable to automatically use every data group in the `conf/data` folder.

The model parameter name for these scripts is the same as the name of the `.yaml` files.

## For fault measuring

To measure the failures of each model instead of their accuracies, you can extract the predicted values of the neural network for the test data.

This can be done by changing the `measure_type` in the configuration to `prediction` from the `accuracy` used in the example. Same goes for script files, where the accuracy needs to be changed in every loop manually, unless you automatize this mechanism.

The correct results are found in the given `test.tsv` files under `conf/data/<data-group-name>` to compare it with the predicted ones. After you collected and compared them you can find categories among them, sort the failures into specific types. In my work i did it manually.

## My measured results

I could categorize the failures of all 3 BERT models on person_psor_noun data group, and categorized failures of XLM-RoBERTa and BERT multilingual BERT models on person_psor_noun data group. Here are the results:

person_psor_noun fail types | XLM-RoBERTa | HuBERT | BERT multilingual
---------- | ---------- | ---------- | ----------
foreign word | 1 | 0 | 1
false data | 4 | 3 | 3
another piece of the sentence is another type | 7 | 1 | 3
close context is another type | 4 | 4 | 8
other | 8 | 0 | 9

mood_verb fail types | XLM-RoBERTa | BERT multilingual
---------- | ---------- | ----------
default Ind choosing | 6 | 6
a word can be more types at once | 3 | 3
other piece of sentence is another type | 1 | 0
close context's mood | 2 | 4
lack of helpful words in context | 1 | 1
false data | 3 | 4
other | 1 | 14

### I created charts of these results, that were used in the thesis too

![](person_psor_fail.PNG)

![](mood_verb_fail.PNG)

## Failure examples for some categories

### From person_psor_noun:

failure type | guessed type | type | word | sentence
---------- | ---------- | ---------- | ---------- | ----------
false data | 2 | 1 | munkát | Tovabbi jó földmunkát !
close context is another type | 2 | 1 | referenciaanyagunkba | A www.giveinformbt.hu honlapunkon betekintést nyerhet referenciaanyagunkba .
another piece of the sentence is another type | 2 | 1 | társadalmunknak | Az öntudat már motoszkál benned , de remélhetőleg idővel egy kicsit megnyugszol és hasznos tagja leszel kis magyar társadalmunknak .

### From mood_verb:

failure type | guessed type | type | word | sentence
---------- | ---------- | ---------- | ---------- | ----------
default Ind choice | Ind | Pot | rajzolhatunk | Egy közkedvelt őszi DIY párnahuzat ötlet , melyet a gyerekekkel közösen is rajzolhatunk .
false data | Ind | Imp | köszönöm/köszönjük | A segítségeteket előre is nagyon köszönöm/köszönjük !
context is another mood | Pot | Imp | hazamehetett. | Még sokminden vár rá , de ma legalább hazamehetett. :)
a word can be more than one type | Ind | Imp | újragondolják | A szociális ügyek megoldásának felelősségét a lakosság tradicionálisan a kormányok , egyház , illetve a családok ügyének tekintette , azonban a gazdasági válság sokakat arra kényszerített , hogy újragondolják ezt a nézetet .
