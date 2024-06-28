from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
text=input("Enter a Text: ")

words=word_tokenize(text)

sent=sent_tokenize(text)

stop=set(stopwords.words("english"))

filter=[]

for word in words:
    if word.lower() not in stop:
        filter.append(word)

print("The Words of the given text: ")
print(words)

print("The Sentences of the given text: ")
print(sent)

print("The Filtered Words: ")
print(filter)

print("stemmed words ")
stemmed = [ps.stem(word) for word in filter]
print(stemmed)

print(lemmatizer.lemmatize("worst", pos='a'))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("running", pos='v'))
print(lemmatizer.lemmatize("ate", pos='v'))