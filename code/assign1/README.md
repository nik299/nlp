# Assignment part 2 readme

## packages imported 

all packages imported for this assignment are written in requirements.txt make sure to install them before running

## usage of pickle

pickle package is used to speedup the building of tf-idf scores for all documents after 2nd run so,after running main.py 
for first time 2 files will be created at running location named 'vocab_list.txt' and 'index_df.txt' 

## usage of tqdm and console output

while  the code is running the program will print some statements in console just to indicate which part of code is 
running and how much is progressed 

## sample Instruction

for evaluation
main.py -dataset [insert your data directory] -out_folder [insert your output directory] -cache_path [insert your directory in which program will save all pickle files] -baseline [put True if you want to get assignment 2 reading] 

note: 
1. if -cache_path is not mentioned program will save in directory in which file is executed
2. -baseline default is False
3. donot add any / or \ after the last directory use D:\PycharmProjects\nlp\cranfield instead of D:\PycharmProjects\nlp\cranfield\


