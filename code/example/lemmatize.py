import spacy
from nltk.stem import SnowballStemmer
import nltk
from sentenceSegmentation import SentenceSegmentation

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# run 'python -m spacy download en' if you have installed for the first time
nlp = spacy.load('en')
document = u"I have a meeting with teacher for dance classes. I am meeting him around 12 noon. Now we are " \
           u"dancing. corpora is plural form of corpus. Apple is singular form of apples ." \
           u" This is same for leaf and leaves "

sb = SnowballStemmer('english')
wn = WordNetLemmatizer()
ss = SentenceSegmentation()
for sent in ss.punkt(document):
    doc = nlp(sent)
    print('spacy -', [token.lemma_ for token in doc])
    print('stemming -', [sb.stem(token.text) for token in doc])
    print('wordnet -', [wn.lemmatize(token.text) for token in doc])
