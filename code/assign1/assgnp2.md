
1.<a>
    Inverted index
    
    cat : docA,docB,docC

    dog : docA,docC

    animal : docB,docC,docA

2.<doubt>
    note: 
    tf-idf is bound to fail since there are 2 words which have representation in all documents

    docA = 0(cat) + 1.216(dog) + 0(animal)

    docB = 0(cat) + 0(dog) + 0(animal)

    docC = 0(cat) + 1.216(dog) + 0(animal)
    
    qcat = 0
    
    qdog = 0.4054(dog)
    
    qanimal = 0

3.<c>
    docA and docC would be retrieved
    
4.<d>

    1 for both docA and docC
    0 for docB
    
5.<5>
    done in informationRetrieval.py
    
6.<6>
    
    a





