from nltk.tokenize import punkt
import json


# Add your import statements here


class SentenceSegmentation():

    def naive(self, text):
        """
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		text : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

        segmented_text = [a.strip(' ') for a in text.replace('? ', '? <>').replace('. ', '. <>').split('<>')]
        if '' in segmented_text:
            segmented_text.remove('')

        # Fill in code here

        return segmented_text

    def punkt(self, text):
        """
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		text : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
        sent_splitter = punkt.PunktSentenceTokenizer()
        segmented_text = sent_splitter.tokenize(text)

        # Fill in code here

        return segmented_text


# this part is used to test the above functions
if __name__ == "__main__":
    queries_json = json.load(open(r"D:\PycharmProjects\nlp\cranfield\cran_queries.json", 'r'))[:]
    queries = [item["query"] for item in queries_json]
    count = 0
    segmenter = SentenceSegmentation()
    for query in queries:
        naive_res = segmenter.naive(query)
        punkt_res = segmenter.punkt(query)
        if naive_res != punkt_res:
            count += 1
            # print(naive_res)
            # print(punkt_res)
    print('ratio of not matched for queries:' + str(count) + '/' + str(len(queries)))
    docs_json = json.load(open(r"D:\PycharmProjects\nlp\cranfield\cran_docs.json", 'r'))[:]
    bodies = [item["body"] for item in docs_json]
    count_body = 0
    for body in bodies:
        naive_res = segmenter.naive(body)
        punkt_res = segmenter.punkt(body)
        if naive_res != punkt_res:
            count_body += 1
            '''
            for i in range(min(len(naive_res), len(punkt_res))):
                if naive_res[i] != punkt_res[i]:
                    if len(naive_res[i]) < len(punkt_res[i]):
                        print(bodies.index(body), naive_res[i])
                        print(punkt_res[i])
                    break
            '''
    print('ratio of not matched for document bodies:' + str(count_body) + '/' + str(len(bodies)))
