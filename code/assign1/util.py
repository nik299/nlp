# Add your import statements here
import numpy as np


# Add any utility functions here

def reslist(query_id, qrels):
    """
    this function gives a list of relevant doc_ids from qrels for a query
    note the list returned may not be in order
    :param query_id:int
                    id of query for which we need list
    :param qrels: list
                    list of dictionaries in format
                  {'query_num':query_id,'position':relevance rank with 1 being most relevant,'id':docIds]}
    :return: true_doc_IDs : list
    """
    true_doc_IDs = []
    for query_dict in qrels:
        if int(query_dict['query_num']) == query_id:
            true_doc_IDs.append(int(query_dict['id']))
    return true_doc_IDs


def rellist(query_id, qrels):
    """

    :param query_id: int
    :param pred_list:list
    :param qrels:
    :return:
    """
    rel_list = []
    for query_dict in qrels:
        if query_dict['query_id'] == query_id:
            rel_list.append(query_dict)

    return rel_list


def scoremake(pred_list, rel_list):
    """

    :param pred_list:
    :param rel_list:
    :return:
    """
    score_list = []
    for pred_id in pred_list:
        for query_dict in rel_list:
            if query_dict['id'] == pred_id:
                score_list.append(5 - query_dict['position'])
            else:
                score_list.append(0)
    return score_list


def dcg(score_list):
    """

    :param score_list:
    :return:sumcg:float
    """
    sumcg = 0
    for i in range(len(score_list)):
        sumcg += score_list[i] / (np.log2(i + 2))
    return sumcg
