from util import *
import json
import pickle


# Add your import statements here


class Evaluation:

    def queryPrecision(self, query_doc_ids_ordered, query_id, true_doc_IDs, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_ids_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of IDs as integers(check) of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

        precision = len(list(set(query_doc_ids_ordered[:k]).intersection(true_doc_IDs))) / k

        # Fill in code here

        return precision

    def meanPrecision(self, doc_ids_ordered, query_ids, qrels, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

        sum_precision = 0
        for query_no in range(len(query_ids)):
            sum_precision += self.queryPrecision(doc_ids_ordered[query_no], query_ids[query_no],
                                                 doc_id_list(query_ids[query_no], qrels), k)
        meanPrecision = sum_precision / len(query_ids)
        # Fill in code here

        return meanPrecision

    def queryRecall(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_ids_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_ids : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
        try:
            recall = len(list(set(query_doc_ids_ordered[:k]).intersection(true_doc_ids))) / len(true_doc_ids)
        except ZeroDivisionError:
            recall = 0
            print(query_id, true_doc_ids)
        # Fill in code here

        return recall

    def meanRecall(self, doc_ids_ordered, query_ids, qrels, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""
        sum_recall = 0
        for query_no in range(len(query_ids)):
            sum_recall += self.queryRecall(doc_ids_ordered[query_no], query_ids[query_no],
                                           doc_id_list(query_ids[query_no], qrels), k)
        meanRecall = sum_recall / len(query_ids)

        # Fill in code here

        return meanRecall

    def queryFscore(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_ids_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_ids : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
        prec = self.queryPrecision(query_doc_ids_ordered, query_id, true_doc_ids, k)
        recal = self.queryRecall(query_doc_ids_ordered, query_id, true_doc_ids, k)
        if prec == 0 and recal == 0:
            fscore = 0
        else:
            fscore = 2 * ((prec * recal) / (prec + recal))

        # Fill in code here

        return fscore

    def meanFscore(self, doc_ids_ordered, query_ids, qrels, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

        sum_fscore = 0
        for query_no in range(len(query_ids)):
            sum_fscore += self.queryFscore(doc_ids_ordered[query_no], query_ids[query_no],
                                           doc_id_list(query_ids[query_no], qrels), k)
        meanFscore = sum_fscore / len(query_ids)

        # Fill in code here

        return meanFscore

    def queryNDCG(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
        """
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		query_doc_ids_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_ids : list
			The list of dictionaries containing IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

        query_score_list = scoremake(query_doc_ids_ordered[:k], true_doc_ids)
        true_doc_IDs_list = []
        for true_doc_ID_dict in true_doc_ids:
            true_doc_IDs_list.append(int(true_doc_ID_dict['id']))
        ideal_score_list = scoremake(true_doc_IDs_list, true_doc_ids)
        if sorted(ideal_score_list, reverse=True)[0] == 0:
            nDCG = 0
            if k == 10:
                print('query id', query_id)
                print([res['id'] + ':' + res['position'] for res in true_doc_ids])
                print(true_doc_IDs_list)
        else:
            nDCG = dcg(query_score_list) / dcg(sorted(ideal_score_list, reverse=True)[:k])
        if k == 10:
            with open(r"D:\PycharmProjects\nlp\code\assign1\dotp.pkl", "rb") as fp:
                docid = pickle.load(fp)
            queries_json = json.load(open(r"D:\PycharmProjects\nlp\cranfield\cran_queries.json", 'r'))[:]
            query_ids, queries = [item["query number"] for item in queries_json], \
                                 [item["query"] for item in queries_json]
            print('query id', query_id, str(nDCG)[:6])
            print(queries[int(query_id) - 1])
            print([str(res['id']) + ':' + str(res['position'])+':'+str(docid[int(query_id) - 1][int(res['id'])])[:6]
                   for res in true_doc_ids])
            print([str(doc)+':'+str(docid[int(query_id) - 1][int(doc)])[:6] for doc in query_doc_ids_ordered[:k]])
        # Fill in code here
        return nDCG

    def meanNDCG(self, doc_ids_ordered, query_ids, qrels, k):
        """
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
        sum_ndcg = 0
        ndcg_list = []
        for query_no in range(len(query_ids)):
            ndcg = self.queryNDCG(doc_ids_ordered[query_no], query_ids[query_no],
                                  dict_list(query_ids[query_no], qrels), k)
            sum_ndcg += ndcg
            ndcg_list.append(ndcg)
        meanNDCG = sum_ndcg / len(query_ids)
        if k == 10:
            dist_plot(ndcg_list, 50)
        # Fill in code here

        return meanNDCG

    def queryAveragePrecision(self, query_doc_ids_ordered, query_id, true_doc_ids, k):
        """
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		query_doc_ids_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_ids : list
			The list of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
        count = 1
        sum_precision = 0
        for i in range(k):
            try:
                if query_doc_ids_ordered[i] in true_doc_ids:
                    count += 1
                    sum_precision += self.queryPrecision(query_doc_ids_ordered, query_id, true_doc_ids, i + 1)
            except IndexError:
                count = k
                break

        avgPrecision = sum_precision / count

        # Fill in code here

        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries
		q_rels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

        sum_avergeprecision = 0
        for query_no in range(len(query_ids)):
            sum_avergeprecision += self.queryAveragePrecision(doc_IDs_ordered[query_no], query_ids[query_no],
                                                              doc_id_list(query_ids[query_no], q_rels), k)
        meanAveragePrecision = sum_avergeprecision / len(query_ids)

        # Fill in code here

        return meanAveragePrecision
