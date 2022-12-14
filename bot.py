import nltk
from gensim.models import Word2Vec
import pandas as pd
import logging


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from clean_normalize_text import clean_sent


# Step 3 Modelisation word2vec

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



model = Word2Vec(
    min_count= 10,# minimum word occurence 
    vector_size = 300, # number of dimensions
    alpha = 0.01, #The initial learning rate
)
cleaned_sentences = clean_sent()
# print(cleaned_sentences)
model.build_vocab(cleaned_sentences)
model.train(cleaned_sentences, total_examples = model.corpus_count, epochs = 60)



def wv_to_df(model):
    all_wv = model.wv.vectors
    
    df = pd.DataFrame(
        all_wv,
        index = model.wv.key_to_index,
        columns = ["dim" + str(i+1) for i in range(all_wv.shape[1])]
    )
    return df

df = wv_to_df(model)
df["idx"] = df.index
print(df.head())

