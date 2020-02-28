import spacy
from nltk.stem import SnowballStemmer
import nltk
from sentenceSegmentation import SentenceSegmentation

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# run 'python -m spacy download en' if you have installed for the first time
nlp = spacy.load('en')
document = u"doctor operates on patient.Kidney operation is important. operational procedure. leaves and leaf"

sb = SnowballStemmer('english')
wn = WordNetLemmatizer()
ss = SentenceSegmentation()
for sent in ss.punkt(document):
    doc = nlp(sent)
    print('spacy -', [token.lemma_ for token in doc])
    print('stemming -', [sb.stem(token.text) for token in doc])
    print('wordnet -', [wn.lemmatize(token.text) for token in doc])
