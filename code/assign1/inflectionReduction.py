from util import *
from nltk.stem import SnowballStemmer


# Add your import statements here


class InflectionReduction:
    def is_number(self, s):
        try:
            float(s)
            return 'num11'
        except ValueError:
            try:
                float(s.replace(',', '.'))
                return 'num11'
            except ValueError:
                return s

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

		we are using snowball stemmer
		"""

        sb = SnowballStemmer('english')
        reducedText = [[self.is_number(sb.stem(word).replace('/', '').replace('-', '')) for word in sentence]
                       for sentence in text]

        # Fill in code here

        return reducedText
