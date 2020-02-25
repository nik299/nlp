
###tokenizers

source link for penn and punkt: http://www.nltk.org/api/nltk.tokenize.html
####Penn Treebank Tokenizer

The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank. This implementation is a port of the tokenizer sed script written by Robert McIntyre and available at [here](http://www.cis.upenn.edu/~treebank/tokenizer.sed. )

it's a top down approach based on some rules.

####Punkt Sentence Tokenizer

This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

The NLTK data package includes a pre-trained Punkt tokenizer for English.

it's a bottom up approach  

###stemming vs lemmatization

source link: https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8

####stemming

best option in nltk is snowball stemmer

####lemmatization 

termed as better since it understands pos but a full sentence is needed for the whole operation ,atleast in spacy