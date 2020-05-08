import nltk
import numpy as np
from tqdm import tqdm
import random
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download('punkt')


class vectorizer:

    def __init__(self):
        self.index = None
        self.docIDs = None
        self.docs = None
        self.queries = None
        self.model = None
        self.vec_size = 200
        self.window = 12
        self.epochs = None
        self.tagged_data = None

    def build_index(self, text, document_ids, epochs):
        self.docIDs = document_ids
        self.docs = text
        self.epochs = epochs

    def rank(self, queries, pre_docs):
        self.queries = queries
        predocint = [[int(x) - 1 for x in pre_doc] for pre_doc in pre_docs]
        doc_IDs_ordered = []
        self.tagged_data = []
        for index, doc in enumerate(self.docs):
            self.tagged_data.append(TaggedDocument(words=doc, tags=['d' + str(self.docIDs[index])]))
        for index, doc in enumerate(self.queries):
            self.tagged_data.append(TaggedDocument(words=doc, tags=['q' + str(index)]))
        max_epochs = 10
        alpha = 0.0025
        random.shuffle(self.tagged_data)
        self.model = Doc2Vec(dm=0, dbow_words=1, vector_size=self.vec_size, window=self.window,
                             min_count=1,
                             )

        self.model.build_vocab(self.tagged_data)

        for epoch in tqdm(range(self.epochs)):
            random.shuffle(self.tagged_data)
            self.model.train(self.tagged_data,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
        self.index = np.zeros((len(self.model.docvecs['d1']), len(self.docs)))
        print('building index')
        for para_index in tqdm(range(len(self.docs))):
            vector = np.array(self.model.docvecs['d' + str(self.docIDs[para_index])])
            self.index[:, para_index] = vector / np.sqrt(vector.dot(vector))

        for qind in tqdm(range(len(self.queries))):

            vector = self.model.docvecs['q'+str(qind)]
            vector = vector / np.sqrt(vector.dot(vector))
            # print(np.amax(np.array(vector).dot(self.index)))
            ind1 = []
            sim_arr = list(vector.dot(self.index[:, predocint[qind][:19]]))
            for i in sorted(sim_arr, reverse=True)[:10]:
                ind1.append(int(predocint[qind][sim_arr.index(i)]) + 1)
            # ind1 = self.docIDs[np.vectorize(int)(np.argpartition(np.array(vector).dot(self.index), -15)[-15:])]
            doc_IDs_ordered.append(ind1)
            print('given', predocint[qind][:10])
            print('result', ind1)

        return doc_IDs_ordered
