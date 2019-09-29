import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

n = 30
filename = 'results-20190921-122938 modified.csv'


def word_processing(sentence):
    results = ''

    for token in sentence:
        if len(token) > 3:
            results = results + " " + str(token.lemma_)
    return results


df = pd.read_csv(filename, delimiter=',')
text = df['titlebody'].values.astype('U')
str_text = []
doc_list = []
words = []
features = []

for i in text:
    i = str(i)
    str_text.append(i)

nlp = spacy.load('en_core_web_sm')

for i in str_text:
    doc = nlp(i)
    doc_list.append(doc)

for i in doc_list:
    words.append(word_processing(i))

tv = TfidfVectorizer(stop_words='english')
tv.fit(words)
filtered_words = tv.transform(words)
features = tv.get_feature_names()

df_tf_idf = pd.DataFrame(filtered_words.toarray()[n], index=features, columns=['TfIdf Score'])
df_tf_idf.sort_values(by=['TfIdf Score'])

values = df_tf_idf['TfIdf Score'].nlargest(3)

print(values)