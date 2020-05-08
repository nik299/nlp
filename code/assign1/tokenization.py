from util import *
import json
from sentenceSegmentation import SentenceSegmentation
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
# Add your import statements here


class Tokenization():

    def naive(self, text):
        """
		Tokenization using a Naive Approach

		Parameters
		----------
		text : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        pattern = r'\s+'
        regexp = RegexpTokenizer(pattern, gaps=True)
        tokenizedText = []
        for a in text:
            tokenizedText += regexp.tokenize(a.replace(',', ' , '))

        # Fill in code here

        return tokenizedText

    def pennTreeBank(self, text):
        """
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		text : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
        penn = TreebankWordTokenizer()
        tokenizedText = []
        for a in text:
            tokenizedText += penn.tokenize(a)

        # Fill in code here

        return tokenizedText


if __name__ == "__main__":
    queries_json = json.load(open(r'D:\PycharmProjects\nlp\cranfield\cran_queries.json', 'r'))[:]
    segmenter = SentenceSegmentation()
    segmented_queries = [segmenter.naive(item["query"]) for item in queries_json]
    count = 0
    tokenizer = Tokenization()
    for query in segmented_queries:
        naive_res = tokenizer.naive(query)
        punkt_res = tokenizer.pennTreeBank(query)
        if naive_res != punkt_res:
            count += 1
        #   print(naive_res)
        #   print(punkt_res)
        #   print(query)
    print('ratio of not matched for segmented_queries:' + str(count) + '/' + str(len(segmented_queries)))
    docs_json = json.load(open(r'D:\PycharmProjects\nlp\cranfield\cran_docs.json', 'r'))[:]
    segmented_bodies = [segmenter.naive(item["body"]) for item in docs_json]
    count_body = 0
    for body in segmented_bodies:
        naive_res = tokenizer.naive(body)
        punkt_res = tokenizer.pennTreeBank(body)
        if naive_res != punkt_res:
            count_body += 1
    #        get some examples if we need to analyse
            '''
            if count_body < 5:
                for i in range(len(naive_res)):
                    if naive_res[i] != punkt_res[i]:
                        print(naive_res[i])
                        print(punkt_res[i])
                    print(body[i])
            '''
    print('ratio of not matched for document segmented_bodies:' + str(count_body) + '/' + str(len(segmented_bodies)))
