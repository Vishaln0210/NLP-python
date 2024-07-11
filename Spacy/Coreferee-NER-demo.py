import spacy
from spacy import displacy
from textblob import TextBlob
from transformers import pipeline
# Get user input text
text = input("Enter a Text: ")

# Process the text
doc = nlp(text)

# Extract entities and relations
relations = []

print("Word In The Text: ")
for token in doc:
    print(token.text)

print("StopWords In The Text: ")
for token in doc:
    if not token.is_stop:
        print(token.text)

print("Lemma In The Text: ")
print("Token : Lemma")
for token in doc:
    print(f"{token.text} : {token.lemma_}")

print("POS In The Text: ")
print("Token : POS : Explanation")
for token in doc:
    pos = token.pos_
    tag = spacy.explain(token.tag_)
    print(f"{token.text} : {pos} : {tag}")

print("NER In The Text: ")
print("Entity : Label : Explanation")
for ent in doc.ents:
    print(f"{ent.text} : {ent.label_} : {spacy.explain(ent.label_)}")

print("Relation in Text:")
for sent in doc.sents:
    for ent in sent.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "NORP"]:
            for token in sent:
                if token.dep_ in ["attr", "nsubj", "dobj", "pobj", "conj", "appos", "acl", "relcl"]:
                    relations.append((ent.text, token.text, sent.text))

for name, rela, context in relations:
    print(f"Relation: {name} -> {rela}")
    print(f"Explanation: '{name}' is related to '{rela}' in the context of '{context}'\n")

context = text + "\n\n"
for _, _, ctx in relations:
    context += ctx + "\n"

print("Coreference Resolution In The Text: ")
coref_chains = doc._.coref_chains
for chain in coref_chains:
    print(f"Chain: {chain}")
    for mention in chain:
        print(f" - Mention: {mention}")

print("Sentiment In The Text: ")
blob = TextBlob(doc.text)
sentiment = blob.sentiment
print(f"Sentiment : {sentiment}")

def answer_question(question, context):
    result = q_pipe(question=question, context=context)
    return result['answer']

question = input("Enter a Question: ")
answer = answer_question(question, text)
print(f"Answer: {answer}")

displacy.serve(doc, style="ent")
