import spacy


class tokenizerLemmatizer:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.list1 = ['aerodynamic', 'approximate', 'axial', 'blunt', 'boundary', 'certain', 'circular', 'compressible',
                      'constant', 'different', 'dimensional', 'exact', 'experimental', 'first', 'flat', 'free',
                      'general', 'good', 'high', 'hypersonic', 'incompressible', 'large', 'local', 'low', 'maximum',
                      'normal', 'numerical', 'particular', 'possible', 'present', 'several', 'similar', 'simple',
                      'small', 'steady', 'subsonic', 'supersonic', 'theoretical', 'thin', 'turbulent', 'uniform',
                      'various', 'viscous']

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
            sent = sent.replace('/', '').replace('(', '').replace(')', '')
            reduced_sent = []
            root = '@@'
            subject = '@@'
            doc = self.nlp(sent)
            for token in doc:
                if token.dep_ != 'punct':
                    reduced_sent.append(self.is_number(token.lemma_.replace('/', '').replace('-', '')))
                '''
                if token.dep_ == 'nsubj':
                    subject = token.lemma_
                if token.dep_ == 'ROOT':
                    root = token.lemma_
            if root != '@@' and subject != '@@':
                reduced_sent.append(root + ' ' + subject)
            '''
            for phrase in doc.noun_chunks:
                for token in phrase:
                    if token.pos_ == 'ADJ' and token.text in self.list1:
                        for token1 in phrase:
                            if token1.pos_ == 'NOUN':
                                reduced_sent.append(token.lemma_+' '+token1.lemma_)
            reduced_text.append(reduced_sent)
        return reduced_text
