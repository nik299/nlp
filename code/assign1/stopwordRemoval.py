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
        stopwordRemovedText = []
        for w in text:
            if w.split('^_*')[0] not in stop_words:
                stopwordRemovedText.append(w.split('^_*')[1])
        # Fill in code here
        stopwordRemovedBigrams =[]
        bi1 = ''
        bi2 = ''
        bi3 = ''
        
        for wo in stopwordRemovedText:
            if bi2 != '':
                bi1 = bi2
                bi2 = bi3
                bi3 = wo
                stopwordRemovedBigrams.append(bi1 + ' ' + bi3)
                # stopwordRemovedBigrams.append(bi2 + ' ' + bi1)
            else:
                bi2 = bi3
                bi3 = wo
        return stopwordRemovedText
