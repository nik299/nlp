# Add your import statements here
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def dummy(doc):
    """
    used for sklearn place holder
    :param doc:
    :return: same as doc
    """
    return doc


# Add any utility functions here

def doc_id_list(query_id, qrels):
    """
    this function gives a list of relevant doc_ids from qrels for a query
    note the list returned may not be in order of importance
    :param query_id: int
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


def dict_list(query_id, qrels):
    """
    this function gives list of dictionaries which are relevant to a given query
    :param query_id: int
                    query no for which we are intrested
    :param qrels: list of dict
                    obtained from qrels.json
    :return: rel_list: list of dict
                    subset of qrels which have re for a single query
    """
    rel_list = []
    for query_dict in qrels:
        if int(query_dict['query_num']) == query_id:
            rel_list.append(query_dict)

    return rel_list


def scoremake(pred_list, rel_list):
    """
     this is a helper function used to covert scores in qrels.json to 5,4,3,2,1,0 format with 5 being most relevant
     and 0 being not relevant
    :param pred_list:list of int
                        list of docIDs for which score is needed
    :param rel_list: list of dict
                        list of dictionaries for a given query
    :return: a list in which ith entry is score of ith docID in pred_list
    """
    score_list = []
    for pred_id in pred_list:
        p = 0
        for query_dict in rel_list:
            if int(query_dict['id']) == pred_id:
                p = 1
                score_list.append(5 - int(query_dict['position']))
                break
        if p == 0:
            score_list.append(0)
    return score_list


def dcg(score_list):
    """
    performs dcg on a list of scores assuming they are in ascending  order
    :param score_list: list of int
    :return:sumcg:float
    """
    sumcg = 0
    for i in range(len(score_list)):
        sumcg += (((2 ** score_list[i]) - 1) / (np.log2(i + 2)))
    #    if sumcg == 0 or np.isnan(sumcg) or np.isinf(sumcg):
    #        print(score_list, sumcg)
    return sumcg


def retokenize(docs):
    """

    :param docs:
    :return:
    """
    done_docs = []
    for doc in docs:
        done_doc = ''
        for sent in doc:
            done_sent = " ".join(str(x) for x in sent) + " "
            done_doc = done_doc.join(done_sent)
        done_docs.append(done_doc)
    return done_docs


def word_pool(docs):
    """
    pools all the words in a list for each sentence to a list for each doc for all docs
    :param docs: a list of docs in which each doc is a list of sentences which are lists of words
    :return: a list of docs in which each doc is a list of words
    """
    new_docs = []
    for doc in docs:
        new_doc = []
        for sentence in doc:
            new_doc += sentence
        new_docs.append(new_doc)
    return new_docs


def dist_plot(x, bins):
    # Plot Histogram on x
    y = np.array(x)
    sns.distplot(y).get_figure().savefig(r"D:\PycharmProjects\nlp\output\query_plot.png")
