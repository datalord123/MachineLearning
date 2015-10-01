from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

vectorizer = CountVectorizer()

string1 = 'hi Katie the self driving car will be late Best Sebastian'
string2 = 'Hi Sebastian the machine learning class will be great great great Best Katie'
string3 = 'Hi Katie the machine learning class will be most excellent'

email_list = [string1,string2,string3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)
#print bag_of_words

#Will return bunch of tuples and integers. Way to interpret '(1,7)   1' for instance,
# would be 'in document 1, word number 7' occurs 1 time.

#print vectorizer.vocabulary_.get('great')
#Returns the position of the word 'great'

sw = stopwords.words('english')
nltk.download()
#sw[0]