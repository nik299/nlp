import spacy
import numpy as np
from tqdm import tqdm


class vectorizer:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.index = None
        self.docIDs = None

    def build_index(self, text, docIDs):
        self.docIDs = docIDs
        assert len(text) == len(docIDs)
        self.index = np.zeros((300, len(text)))
        print('building index')
        for para_index in tqdm(range(len(text))):
            doc = self.nlp(text[para_index])
            if doc.vector_norm != 0:
                self.index[:, para_index] = doc.vector / doc.vector_norm

    def rank(self, queries):
        doc_IDs_ordered = []

        for query in tqdm(queries):
            doc = self.nlp(query)
            vector = np.array(doc.vector / doc.vector_norm)

            sim_arr = list(np.array(vector).dot(self.index))
            # print(np.amax(sim_arr))
            ind1 = []
            ind2 = []
            for i in sorted(sim_arr, reverse=True)[:15]:
                ind1.append(self.docIDs[sim_arr.index(i)])
                ind2.append(dict({sim_arr.index(i):i}))
            # ind1 = self.docIDs[np.vectorize(int)(np.argpartition(np.array(vector).dot(self.index), -15)[-15:])]
            print(ind2)
            doc_IDs_ordered.append(ind1)

        return doc_IDs_ordered
