import spacy
from nltk.wsd import lesk


class tokenizerLemmatizer:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.list1 = ['aerodynamic', 'approximate', 'axial', 'blunt', 'boundary', 'certain', 'circular', 'compressible',
                      'constant', 'different', 'dimensional', 'exact', 'experimental', 'first', 'flat', 'free',
                      'general', 'good', 'high', 'hypersonic', 'incompressible', 'large', 'local', 'low', 'maximum',
                      'normal', 'numerical', 'particular', 'possible', 'present', 'several', 'similar', 'simple',
                      'small', 'steady', 'subsonic', 'supersonic', 'theoretical', 'thin', 'turbulent', 'uniform',
                      'various', 'viscous']
        self.pos_dict = dict(
            {"": None, "ADJ": 'a', "ADP": None, "ADV": 'r', "AUX": None, "CONJ": None, "CCONJ": None, "DET": None,
             "INTJ": None,
             "NOUN": 'n',
             "NUM": None,
             "PART": None,
             "PRON": 'n',
             "PROPN": 'n',
             "PUNCT": None,
             "SCONJ": None,
             "SYM": None,
             "VERB": 'v',
             "X": None,
             "EOL": None,
             "SPACE": None
             })

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

    def spacy_lemmatizer(self, docs):
        reduced_text = []
        for sent in docs:
            sent = sent.replace('/', '').replace('(', '').replace(')', '')
            reduced_sent = []
            root = '@@'
            subject = '@@'
            procesed_sent = self.nlp(sent)
            bi1 = ''
            bi2 = ''
            for token in procesed_sent:
                if token.dep_ != 'punct':
                    uni_word = self.is_number(token.lemma_.replace('/', '').replace('-', ''))
                    try:
                        uni_word = uni_word + '^_*' + lesk(sent, uni_word).name()
                    except AttributeError:
                        try:
                            uni_word = uni_word + '^_*' + lesk(uni_word, uni_word).name()
                        except AttributeError:
                            uni_word = uni_word + '^_*' + uni_word
                    reduced_sent.append(uni_word)

                '''
                if token.dep_ == 'nsubj':
                    subject = token.lemma_
                if token.dep_ == 'ROOT':
                    root = token.lemma_
            if root != '@@' and subject != '@@':
                reduced_sent.append(root + ' ' + subject)
     
            
            for phrase in doc.noun_chunks:
                for token in phrase:
                    if token.pos_ == 'ADJ' and token.text in self.list1:
                        for token1 in phrase:
                            if token1.pos_ == 'NOUN':
                                reduced_sent.append(token.lemma_ + ' ' + token1.lemma_)
            if len(reduced_sent) == 0:
                reduced_sent.append('place_holder')
            '''

            reduced_text += reduced_sent
        return reduced_text
