
1.<a>
    Inverted index
    
    cat : docA,docB,docC

    dog : docA,docC

    animal : docB,docC,docA

2.<doubt>
    note: 
    tf-idf is bound to fail since there are 2 words which have representation in all documents

    docA = 0(cat) + 3*log(3/2)(dog) + 0(animal)

    docB = 0(cat) + 0(dog) + 0(animal)

    docA = 0(cat) + 3*log(3/2)(dog) + 0(animal)

3.<c>
    docA and docC would be retrieved
    
4.<d>
    




