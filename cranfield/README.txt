The Cranfield dataset contains 1400 documents (cran_docs.json), 225 queries (cran_queries.json), and query-document relevance judgements (cran_qrels.json).

The positions of the reference documents for each query (in cran_qrels.json) indicate the following judgements:
1 - References which are a complete answer to the question.
2 - References of a high degree of relevance, the lack of which either would have made the research impracticable or would have resulted in a considerable amount of extra work.  
3 - References which were useful, either as general background to the work or as suggesting methods of tackling certain aspects of the work.
4 - References of minimum interest, for example, those that have been included from an historical viewpoint.
Query-Reference pairs in which the reference is of no interest to the query are excluded from the relevance file.



More on the Cranfield dataset: http://ir.dcs.gla.ac.uk/resources/test_collections/cran/


This folder contains the additional files required for Part 2 of the assignment, involving building a search engine application. Note that this code works for both Python 2 and Python 3.

The following files have been added:
informationRetrieval.py and evaluation.py - Implement the corresponding tasks inside the functions in these files.

The following file has been updated:
main.py - The main module that contains the outline of the Search Engine. It has been updated to include calls to the information retrieval and evaluation tasks, in addition to the tasks solved in Part 1 of the assignment. Do not change anything in this file.

For this part of the assignment, you are advised to make a copy of the completed code from Part 1 of the assignment - replace the main file with the updated version and add and fill in the new files (informationRetrieval.py and evaluation.py).

To test your code, run main.py as before with the appropriate arguments.
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)]

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.