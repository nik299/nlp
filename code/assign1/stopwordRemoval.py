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
		stopword_removed_synsets : list
			list is a sequence of tokens
			representing a document with stopwords removed
		"""

        stop_words = set(['', ' ', '  ', "'", "'s", '.', '=', 'PRON', 'a.'])
        stopword_removed_synsets = []
        for sentence in text:
            stopword_removed_sentence = []
            for word_synset in sentence:
                if word_synset.split('^_*')[0] not in stop_words:
                    stopword_removed_sentence.append(word_synset.split('^_*')[0])
            stopword_removed_bigrams = []
            bi2 = ''

            for wo in stopword_removed_sentence:
                if bi2 != '':
                    bi1 = bi2
                    bi2 = wo
                    stopword_removed_bigrams.append(bi1 + ' ' + bi2)
                    # stopword_removed_bigrams.append(bi2 + ' ' + bi1)
                else:
                    bi2 = wo
            stopword_removed_synsets.append(stopword_removed_sentence)
            #stopword_removed_synsets.append(stopword_removed_bigrams)
        # Fill in code here
        '''
        stopword_removed_bigrams =[]
        bi1 = ''
        bi2 = ''
        bi3 = ''
        
        for wo in stopword_removed_synsets:
            if bi2 != '':
                bi1 = bi2
                bi2 = bi3
                bi3 = wo
                stopword_removed_bigrams.append(bi1 + ' ' + bi3)
                # stopword_removed_bigrams.append(bi2 + ' ' + bi1)
            else:
                bi2 = bi3
                bi3 = wo
        '''
        return stopword_removed_synsets
