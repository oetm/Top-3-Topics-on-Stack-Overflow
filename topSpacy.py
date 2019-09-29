import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

n = 3
no_top_words = 10


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % idx)
        print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])


def word_processing(text):
    results = []

    for token in text:
        if len(token) > 3:
            results.append(token.lemma_)

    return results


df = pd.read_csv('results-20190921-122938.csv', delimiter=',')
text = str(df['title'][0:10].values.astype('U'))

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

words = word_processing(doc)

cv = CountVectorizer(analyzer='word',lowercase=True, stop_words='english')
filtered_words = cv.fit_transform(words)
features = str(cv.get_feature_names())

lda = LatentDirichletAllocation(n_components=n, max_iter=20, learning_method='online')
lda_data = lda.fit(filtered_words)

nmf_model = NMF(n_components=n)
nmf_Z = nmf_model.fit(filtered_words)

print_topics(lda, cv)
print_topics(nmf_model, cv)