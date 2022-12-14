import nltk
from nltk import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



nltk.download('punkt')


from pdf_to_txt import transform_pdf_to_text




text = transform_pdf_to_text()
sentences = sent_tokenize(text)

# Step 2 Clean and normalize data 
def clean_text(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    tokens = word_tokenize(text)
    # Remove the punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("french")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tokens


def remove_stopwords(text, stopw = stopwords.words("french")):
    list_of_sentences = []
    
    for sentences in text:
        list_of_words = []
        for word in sentences:
            if not word in stopw:
                list_of_words.append(word)
        list_of_sentences.append(list_of_words)
    return list_of_sentences

tokenizer = RegexpTokenizer(r'\w+')

def clean_sent(sentences=sentences):
    """Sentence must be a list containing string"""
    stopw = stopwords.words("french")
    # Lower each word in each sentence        
    sentences = [tokenizer.tokenize(sent.lower()) for sent in sentences]
    sentences = remove_stopwords(sentences)
    return sentences

# print(clean_sent())