import spacy
from spacy.lang.en import English
from spacy.lang.en.examples import sentences

# nlp = English()
nlp = spacy.load('en_core_web_sm')
for row, doc in enumerate(nlp.pipe(sentences)):
    print(doc.text)
    print(doc.to_array([spacy.attrs.LEMMA, spacy.attrs.LIKE_EMAIL]))
    for token in doc:
        print(token.text, token.pos_, token.dep_)
