from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(dc):
	stop_free = ' '.join([i for i in dc.lower().split() if i not in stop])
	punc_free = ''.join([cha for cha in stop_free if cha not in exclude])
	normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
	remove_dig = re.sub(r"\d+","",normalized)
	y = remove_dig.split()
	return y

print("There are 10 setences of following three topics")
path = "d:/Sentences.txt"
train_sentences = []
fin = open(path, 'r')
for line in fin:
	line = line.strip();
	cleaned = clean(line)
	cleaned = ' '.join(cleaned)
	train_sentences.append(cleaned)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_sentences)
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X, y_train)

modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)

path_test = "d:/test_s.txt"
fin = open(path_test,'r')
test_sentences = fin.readlines()
test_clean_sentence = []
for test in test_sentences:
	cleaned_test = clean(test)
	cleaned = ' '.join(cleaned_test)
	cleaned = re.sub(r"\d+","",cleaned)
	test_clean_sentence.append(cleaned)
Test = vectorizer.transform(test_clean_sentence)

labels = ['Cricket','AI','Che']
predicted_knn = modelknn.predict(Test)
predicted_kmeans = modelkmeans.predict(Test)
answer_nlp = "d:/answer_nlp.txt"
fout = open(answer_nlp,'w')
for i in range(len(test_sentences)):
	fout.writelines([test_sentences[i],":",labels[np.int(predicted_knn[i])],'\n'])
lab_dict = {Counter(modelkmeans.labels_[0:10]).most_common(1)[0][0]:'Cricket',
			Counter(modelkmeans.labels_[10:20]).most_common(1)[0][0]:'AI',
			Counter(modelkmeans.labels_[20:30]).most_common(1)[0][0]:'Che'}
for i in range(len(test_sentences)):
	fout.writelines([test_sentences[i],":",lab_dict[predicted_kmeans[i]],'\n'])


