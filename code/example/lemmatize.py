import spacy
from nltk.stem import SnowballStemmer

# run 'python -m spacy download en' if you have installed for the first time
nlp = spacy.load('en')
doc = nlp(u"effects of extreme surface cooling on boundary layer transition .   an investigation was made to "
          u"determine the combined effects of surface cooling, pressure gradients, nose blunting, and surface finish "
          u"on boundary-layer transition .  data were obtained for various body shapes at a mach number of 3.12 and "
          u"reynolds numbers per foot as high as 15x10 .   previous transition studies, with moderate cooling, "
          u"have shown agreement with the predictions of stability theory .  for surface roughnesses ranging from 4 "
          u"to 1250 microinches the location of transition was unaffected with moderate cooling .  with extreme "
          u"cooling, an adverse effect was observed for each of the parameters investigated .  in general, "
          u"the transition reynolds number decreased with decreasing surface temperature . in particular, "
          u"the beneficial effects of a favorable pressure gradient obtained with moderate cooling disappear with "
          u"extreme cooling, and a transition reynolds number lower than that observed on a cone is obtained . "
          u"further, an increase in the nose bluntness decreased the transition reynolds number under conditions of "
          u"extreme cooling .")
sb = SnowballStemmer('english')
print([token.lemma_ for token in doc])
print([sb.stem(token.text) for token in doc])
