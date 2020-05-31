from nltk.wsd import lesk


class leskTeat():

    def synsetmake(self,docs):
        synsetdocs =[]
        for sent in docs:
            for word in sent:
                synsetdocs.append(str(word) + '^_*' + str(lesk(sent,word)) )

        return synsetdocs