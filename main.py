import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

df = pd.read_csv(r'50kurls_7days.csv').head(10000)

# Removing protocols and domains from URLs
corpus_url_free=[]
for url in df['SITE_PAGE']:
    a = re.sub('https?://(?:[-\w.]|(%[\da-zA-Z]{2}))+','', url)   #used regex substitute to remove protocol and domain.
    corpus_url_free.append(a)

# splitting corpus by'/'
split_corpus=[]
for url in corpus_url_free:
    url = url.split('/')
    x = ' '.join(url).strip()
    split_corpus.append(x)

# Cleaning corpus (removing stopwords,numbers etc.)
corpus = []
for i in range(len(split_corpus)):
    review = re.sub('[^a-zA-Z]',' ',split_corpus[i])
    review=review.replace("html","")
    review=review.replace("php","")
    review = review.lower()
    review = review.split()
    review = [word for word in review if word not in set(stopwords.words('english'))]  # removing stopwords.
    review = ' '.join(review)
    corpus.append(review)

# Vectorisation with ngrame range(1,3)
cv = CountVectorizer(ngram_range=(1,3))

vector = cv.fit_transform(corpus)

# Fetching Keywords from vocab created by countvectorisation.
keywords = cv.get_feature_names_out()

# Converting vectors to array to perform metrics operations.
vector_array = vector.toarray()

# Fetching URL frequency as doc_freq from original dataframe to find popularity score..
doc_freq = df['FREQUENCY']

trans_doc_freq = np.transpose(doc_freq)  #transposed document frequency to get it in shape to mutiply with vector array.

# matrics multiplication of transposed document frequency and vector array which provides popularity score.
popularity_score = np.matmul(trans_doc_freq, vector_array)

# zip keyword and popularity score and convert into dataframe
data=pd.DataFrame(list(zip(keywords,popularity_score)),
               columns =['Keywords', 'Popularity_Score'])

# Sort the dataframe according to popularity column to get the final dataframe.
data1 = data.sort_values(by=['Popularity_Score'],ascending=False).reset_index()    #sorted by popularity score.


## Streamlit app Representation ##

n = st.text_input("Enter keywords",)

suggest = []
for i in data1['Keywords']:
    if i.startswith(n):
        suggest.append(i)

if st.button('Suggestion'):
    st.write(suggest[:10])