# Thesis work

## The reason

* This repository is containing files that were used to work on the project of measuring different BERT models by probing, in order to find differences, and decide which model is the better than the other, and why. The thesis was written in hungarian so some files have hungarian names

## The Jupyter notebooks and the sln file's connection

* I used a remote LINUX computer for some of my works, and since i used jupyter notebooks and some importable libraries, i needed something to import them with. This was Miniconda, that is contained in the sln file.
* For better understanding i used jupyter notebooks first to create the python files, that helped me to analyse the BERT models. I wanted to deep dive into pandas, to use it for the program, so i tried the DataFrames' functions out, by a jupyter notebook of my consultant. 
* Then for preparation, i created a jupyter notebook to measure 2 BERT models' tokenizers and their results manually, so i could see how they work. 
* I used then the hidden layers of the BERT models for probing, so i had to find out how can i extract the given tensors from it. I created another notebook for that purpose. 
* When i knew everything i needed, i created the program, that did the actual probing and showed the results of the training.
* After i created the real probing program in python, and extracted results into python files, then put together into csv files, i used another jupyter notebook file to plot the results for the thesis work, and for myself to see a bigger picture.

## The python files

* The model_test_1.py was the program created in jupyter notebook, that i used to run the same probing, but now saving the results into result files.
* The model_configurable.py is the program that i used to get the final results into result files. It is configurable by hydra, and usable to test any BERT model of the 3 on any data group out of the 11 by any hidden layer of the BERT out of the 13. 

## The conf directory

* This directory is for configuring the program, that was used for measuring the BERT models.
* It contains yaml files for each configuration option, by the hydra standard.
* The subfolders in the data folder contain the actual data files sorted by data group.
* Each data group consists of a train.tsv, a dev.tsv and a test.tsv that contain the data itself in a given structure.

## The scripts

* There are two script files now, script_try and script_try_first_half, these are looking for a model as a parameter, and they will iterate through all data groups and all layers, and save the results into the given folders for more evaluating. They are made very simply, one iterates through the first 7 layers, the other uses the last 6.

## The resultss directory

* This directory contains the results of each BERT model.
* In every BERT model's folder, there are the predicted files for a few data groups. Those were used for categorization of the faults of each BERT model.
* In every BERT model's folder the results with dev and test accuracies are sorted, by the used hidden layer of the BERT, into subfolders.
* The layer subfolders contain a file with every data groups' results in them, with dev and test accuracies.

## The csv, xlsx and pptm files

* These files were used to get a bigger picture of the collected datas. They were created by collecting out every test accuracy from the result files in every subfolder of the results folder.
* The ones that have latex in their names were used in the thesis itself, as tables.
* The charts.xlsx contains the charts that were used in the thesis, and were made by the results of categorization of the faults of each BERT model. It is a not well distributed file, since it was only needed for creating the charts.
* In the pptm file i created figures for the thesis, to represent some parts of it.

## The PNG files and the pdf file

* The PNG files were used for the thesis to make it more readable and easier to imagine some parts in it. The pdf contained the graphs created by the plotting jupyter notebook, and the graph PNGs used this file as source.
* Some PNG files used the charts.xlsx file for source, others used the pptm, others used outer sources, mentioned in the thesis.
