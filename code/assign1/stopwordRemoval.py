from util import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Add your import statements here


class StopwordRemoval():

    def fromList(self, text):
        """
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		text : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		stopwordRemovedText : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

        stop_words = set(stopwords.words('english')+['', ' ', '  ', "'", "'s", '.', '=', 'PRON', 'a.'])
        stopwordRemovedText = [w for w in text if w.split('@@')[0] not in stop_words]

        # Fill in code here

        return stopwordRemovedText
