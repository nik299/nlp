
###tokenizers

source link for penn and punkt: http://www.nltk.org/api/nltk.tokenize.html

####Punkt Sentence Tokenizer

This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

The NLTK data package includes a pre-trained Punkt tokenizer for English.

it's a bottom up approach  

####punkt vs naive approach
a.no example for now to wsay naive is better than punkt
b.example mentioned in question 2 and "a. ferri's vortical layer is brought into evidence ." In this sentence poorly
written grammar at a. ferri's 

####Naive word tokenizer

A naive apprach for word tokenizer is to split along the whitespaces and commas.

####Penn Treebank Tokenizer

The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank. This implementation is a port of the tokenizer sed script written by Robert McIntyre and available at [here](http://www.cis.upenn.edu/~treebank/tokenizer.sed. )

it's a top down approach based on some rules.


###stemming vs lemmatization

source link: https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8

####stemming

best option in nltk is snowball stemmer. Downside to this is it doesn't get the context
eg: meeting as a noun and as a verb both give meet when stemmed

####lemmatization 

termed as better since it understands pos but a full sentence is needed for the whole operation ,at least in spacy. It is not accurate in case if we use wordnet since by default it assumes noun and any correction to it is complex when compared to using spacy.