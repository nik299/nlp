from util import *
import pandas as pd
import numpy as np
from scipy.linalg import svd
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from util import dummy

# Add your import statements here


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.ini_index = None
        self.vocab_list = None
        self.docIDs = None
        self.pipe = None

    def buildIndex(self, docs, docIDs):
        """
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable as a dataframe of tf-idf vectors along with idf scores to each
		word and and also stores vocabulary in vocab_list
		document number used in docIDs

		Parameters
		----------
		docs : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
			note: I am considering document position as document ID's
		docIDs : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
        self.docIDs = docIDs
        try:
            with open(r"/vocab_list.pkl", "rb") as fp:  # Unpickling
                self.vocab_list = pickle.load(fp)
        except IOError or FileNotFoundError:
            self.vocab_list = []
            print('building vocabulary')
            for doc in tqdm(range(len(docs))):
                for word in docs[doc]:
                    if word not in self.vocab_list:
                        self.vocab_list.append(word)
            self.vocab_list.sort()
            with open("vocab_list.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.vocab_list, fp)

        index_df = pd.DataFrame(data=np.zeros((len(self.vocab_list), len(self.docIDs) + 2)), index=self.vocab_list,
                                columns=docIDs + ['n_i', 'idf'])
        try:
            with open("index_df.pkl", "rb") as fp:  # Unpickling
                index_df = pickle.load(fp)
        except IOError or FileNotFoundError:
            print('creating tf-idf vectors for documents')
            for doc_ind in tqdm(range(len(docs))):
                word_list = []
                for word in docs[doc_ind]:
                    if word in list(index_df.index):
                        index_df.loc[[word], [docIDs[doc_ind]]] += 1
                        if word not in word_list:
                            index_df.loc[[word], ['n_i']] += 1
                            word_list.append(word)
            with open("index_df.pkl", "wb") as fp:  # Pickling
                pickle.dump(index_df, fp)
        index_df['idf'] = index_df['n_i'].apply(lambda x: np.log10(len(docs) / x) if x > 0 else 0)
        index_df[docIDs] = index_df[docIDs].mul(index_df['idf'].to_numpy(), axis='rows')

        self.ini_index = index_df.loc[index_df['n_i'] > 2]

    def buildIndex1(self, docs, docIDs):
        self.docIDs = docIDs
        self.pipe = Pipeline([('count', CountVectorizer(tokenizer=dummy, preprocessor=dummy, )),
                         ('tfid', TfidfTransformer())]).fit(docs)
        X = self.pipe['count'].transform(docs)
        df = pd.DataFrame(X.toarray(), columns=self.pipe['count'].get_feature_names())
        index_df = df.transpose()
        index_df.columns = docIDs
        index_df['idf'] = self.pipe['tfid'].idf_
        self.ini_index = index_df

    def lsi(self, num_vec):
        index_arr = self.ini_index.to_numpy()[:, :-1]
        U, s, VT = svd(index_arr)
        Sigma = np.zeros((index_arr.shape[0], index_arr.shape[1]))
        Sigma[:index_arr.shape[1], :index_arr.shape[1]] = np.diag(s)
        Sigma = Sigma[:, :num_vec]
        VT = VT[:num_vec, :]
        # reconstruct
        B = U.dot(Sigma.dot(VT))
        b_df = pd.DataFrame(data=B, index=list(self.ini_index.index), columns=self.docIDs)
        # b_df['n_i'] = self.ini_index['n_i']
        b_df['idf'] = self.ini_index['idf']
        self.index = b_df

    def no_lsi(self):
        self.index = self.ini_index


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
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
        doc_IDs_ordered = []
        query_df = pd.DataFrame(data=np.zeros((len(list(self.index.index)), len(queries))),
                                index=list(self.index.index), columns=range(len(queries)))
        query_df['idf'] = self.index['idf']

        doc_mag = self.index[self.docIDs].mul(self.index[self.docIDs].to_numpy(), axis='rows').sum(axis=0) \
            .apply(np.sqrt)
        print('matching documents with queries')
        for query_ind in tqdm(range(len(queries))):
            for word in queries[query_ind]:
                if word in list(self.index.index):
                    query_df.loc[[word], [query_ind]] += 1
            query_df[query_ind] = query_df[query_ind].mul(query_df['idf'].to_numpy(), axis='rows')
            dot_p = self.index[self.docIDs].mul(query_df[query_ind].to_numpy(), axis='rows').sum(axis=0)
            dot_p = dot_p.div(doc_mag).fillna(0) / np.sqrt(query_df[query_ind]
                                                           .mul(query_df[query_ind].to_numpy(), axis='rows')
                                                           .sum(axis=0))
            doc_IDs_ordered.append(list(dot_p.loc[dot_p > 0].sort_values(ascending=False).index))
        return doc_IDs_ordered
