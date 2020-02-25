from util import *
from nltk.stem import SnowballStemmer


# Add your import statements here


class InflectionReduction:

    def reduce(self, text):
        """
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence

		we are using snowball stemmer for now#TODO
		"""

        sb = SnowballStemmer('english')
        reducedText = [[sb.stem(word) for word in sent] for sent in text]

        # Fill in code here

        return reducedText
