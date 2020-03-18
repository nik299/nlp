from util import *
import pandas as pd
import numpy as np


# Add your import statements here


class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs):
        """
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		docs : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
			note: I am considering document position as document ID's #TODO
		docIDs : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
        vocab_list = []
        for doc in docs:
            for sent in docs:
                vocab_list = vocab_list + sent

        vocab_list = list(set(vocab_list))
        index_df = pd.DataFrame(data=np.zeros((len(vocab_list), len(docIDs) + 2)), index=vocab_list,
                                columns=docIDs + ['n_i', 'idf'])
        for doc in range(len(docs)):
            p = 0
            for sent in docs[doc]:
                for word in sent:
                    if word in vocab_list:
                        index_df[word, doc] += 1
                        if p == 0:
                            index_df[word, 'n_i'] += 1
                            p = 1
        index_df['idf'] = index_df['n_i'].apply(lambda x: np.log(len(docs) / x))

        index = index_df

        # Fill in code here

        self.index = index

    def rank(self, queries):
        """
		Rank the documents according to relevance for each query

		Parameters
		----------
		queries : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

        doc_IDs_ordered = []

        # Fill in code here

        return doc_IDs_ordered
