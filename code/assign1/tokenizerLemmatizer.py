import spacy


class tokenizerLemmatizer:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def is_number(self, s):
        try:
            float(s)
            return 'num11'
        except ValueError:
            try:
                float(s.replace(',', ''))
                return 'num11'
            except ValueError:
                return s

    def spacy_lemmatizer(self, text):
        reduced_text = []
        for sent in text:
            root = '@@'
            subject = '@@'
            doc = self.nlp(sent)
            for token in doc:
                reduced_text.append([self.is_number(token.lemma_.replace('/', '').replace('-', ''))])
                if token.dep_
            if root != '@@' or subject != '@@':
                reduced_text.apppend(root+' '+subject)
        return reduced_text
