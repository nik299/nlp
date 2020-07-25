from sentenceSegmentation import SentenceSegmentation
from tokenizerLemmatizer import tokenizerLemmatizer
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from gensimvect import vectorizer
from sys import version_info
import os
import pickle
from nltk.wsd import lesk
import argparse
import json
import matplotlib.pyplot as plt
from util import word_pool

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, main_args, cache_dir_passed):
        self.cache_dir = cache_dir_passed
        self.args = main_args

        self.tokenizer = Tokenization()
        self.tokenAndLemmatizer = tokenizerLemmatizer()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        self.vectorizer = vectorizer()
        self.informationRetriever = InformationRetrieval(self.args.baseline, self.cache_dir)
        self.evaluator = Evaluation(self.cache_dir, self.args)

    def segmentSentences(self, text):
        """
		Call the required sentence segmenter
		"""
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenizeAndLemmatize(self, text):
        """
        does both tokenizeing and lemmatizing
        """
        return self.tokenAndLemmatizer.spacy_lemmatizer(text)

    def tokenize(self, text):
        """
		Call the required tokenizer
		"""
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
		Call the required stemmer/lemmatizer
		"""
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
		Call the required stopword remover
		"""
        return self.stopwordRemover.fromList(text, self.args.baseline)

    def preprocessQueries(self, queries):
        """
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(os.path.join(self.args.out_folder, "segmented_queries.txt"), 'w'))
        # Tokenize queries
        tokenizedQueries = []
        tokenized_lemmatizedQueries = []  # tokenize and lemmatize  queries
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenized_lemmatizedQuery = self.tokenizeAndLemmatize(query)
            tokenized_lemmatizedQueries.append(tokenized_lemmatizedQuery)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(os.path.join(self.args.out_folder, "tokenized_queries.txt"), 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(os.path.join(self.args.out_folder, "reduced_queries.txt"), 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        if self.args.baseline:
            for query in reducedQueries:
                stopwordRemovedQuery = self.removeStopwords(query)
                stopwordRemovedQueries.append(stopwordRemovedQuery)
        else:
            for query in tokenized_lemmatizedQueries:
                stopwordRemovedQuery = self.removeStopwords(query)
                stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(os.path.join(self.args.out_folder, "stopword_removed_queries.txt"), 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs, title=False):
        """
		Preprocess the documents
		"""

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        if not title:
            json.dump(segmentedDocs[0], open(os.path.join(self.args.out_folder, "segmented_docs.txt"), 'w'))
        # Tokenize docs
        tokenizedDocs = []
        tokenized_lemmatizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenized_lemmatizedDoc = self.tokenizeAndLemmatize(doc)
            tokenizedDocs.append(tokenizedDoc)
            tokenized_lemmatizedDocs.append(tokenized_lemmatizedDoc)
        if not title:
            json.dump(tokenizedDocs[0], open(os.path.join(self.args.out_folder, "tokenized_docs.txt"), 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        if not title:
            json.dump(reducedDocs[0], open(os.path.join(self.args.out_folder, "reduced_docs.txt"), 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        if self.args.baseline:
            for doc in reducedDocs:
                stopwordRemovedDoc = self.removeStopwords(doc)
                stopwordRemovedDocs.append(stopwordRemovedDoc)
        else:
            for doc in tokenized_lemmatizedDocs:
                stopwordRemovedDoc = self.removeStopwords(doc)
                stopwordRemovedDocs.append(stopwordRemovedDoc)
        if not title:
            json.dump(stopwordRemovedDocs[0],
                      open(os.path.join(self.args.out_folder, "stopword_removed_docs.txt"), 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

        # Read queries
        queries_json = json.load(open(os.path.join(self.args.dataset, "cran_queries.json"), 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
                             [item["query"] for item in queries_json]
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(os.path.join(self.args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids, docs, titles = [item["id"] for item in docs_json], \
                                [item["body"] for item in docs_json], [item['title'] for item in
                                                                       docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        processedTitles = self.preprocessDocs(titles, title=True)

        # Build document index
        if self.args.baseline:
            print(len(processedDocs))
            self.informationRetriever.buildIndex1(word_pool(processedDocs), doc_ids)
            self.informationRetriever.no_lsi()
            doc_IDs_ordered = self.informationRetriever.rank(word_pool(processedQueries))
            import pickle
            with open(os.path.join(self.cache_dir, "doc_Ids.pkl"), "wb") as fp:  # Pickling
                pickle.dump(doc_IDs_ordered, fp)

        else:
            self.informationRetriever.buildIndex1(word_pool(processedDocs), doc_ids)
            self.informationRetriever.buildIndex2(word_pool(processedDocs), doc_ids)
            self.informationRetriever.buildTitleIndex1(word_pool(processedTitles))

            self.informationRetriever.sk_lsi(430)
            self.informationRetriever.sk_lsi_bi(320)
            self.informationRetriever.combine(0.2)
            # self.informationRetriever.no_lsi()
            # Rank the documents for each query
            doc_IDs_ordered = self.informationRetriever.rank(word_pool(processedQueries))
            # self.vectorizer.build_index(processedDocs, doc_ids, 20)
            # doc_IDs_ordered = self.vectorizer.rank(processedQueries,doc_IDs_ordered1)
            import pickle
            with open(os.path.join(self.cache_dir, "doc_Ids.pkl"), "wb") as fp:  # Pickling
                pickle.dump(doc_IDs_ordered, fp)
        # Read relevance judgements
        qrels = json.load(open(os.path.join(self.args.dataset, "cran_qrels.json"), 'r'))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        p = 11
        for k in range(1, p):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +
                  str(k) + " : " + str(precision) + ", " + str(recall) +
                  ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
                  str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, p), precisions, label="Precision")
        plt.plot(range(1, p), recalls, label="Recall")
        plt.plot(range(1, p), fscores, label="F-Score")
        plt.plot(range(1, p), MAPs, label="MAP")
        plt.plot(range(1, p), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(os.path.join(self.args.out_folder, "eval_plot.png"))

    def handleCustomQuery(self):
        """
		Take a custom query as input and return top five relevant documents
		"""

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(os.path.join(self.args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids, docs, titles = [item["id"] for item in docs_json], \
                                [item["body"] for item in docs_json], [item['title'] for item in
                                                                       docs_json]
        # Process documents
        try:
            with open(os.path.join(self.cache_dir, "index_df1.pkl"), "rb") as fp:  # Unpickling
                index_df = pickle.load(fp)
            with open(os.path.join(self.cache_dir, "pipeline.pkl"), "rb") as fp:
                pipe = pickle.load(fp)
            processedDocs = [""]
        except IOError or FileNotFoundError:
            processedDocs = self.preprocessDocs(docs)

        try:
            with open(os.path.join(self.cache_dir, "title_index_df1.pkl"), "rb") as fp:  # Unpickling
                index_df = pickle.load(fp)
            processedTitles = [""]
        except IOError or FileNotFoundError:
            processedTitles = self.preprocessDocs(titles)

        # Build document index
        self.informationRetriever.buildIndex1(word_pool(processedDocs), doc_ids)
        self.informationRetriever.buildIndex2(word_pool(processedDocs), doc_ids)
        self.informationRetriever.buildTitleIndex1(word_pool(processedTitles))

        self.informationRetriever.sk_lsi(395)
        self.informationRetriever.sk_lsi_bi(395)
        self.informationRetriever.combine(0.2)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank(word_pool(processedQuery))[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default="cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="output/",
                        help="Path to output folder")
    parser.add_argument('-cache_path', default=os.path.dirname(os.path.abspath(__file__)),
                        help='directory in which you want to place cache')
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer', default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")
    parser.add_argument('-baseline', default=False,
                        help='measure baseline readings(of previous assignment')

    # Parse the input arguments
    args = parser.parse_args()
    cache_dir = os.path.join(args.cache_path, 'cache')
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args, cache_dir)

    # Either handle query from user or evaluate on the complete dataset
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
