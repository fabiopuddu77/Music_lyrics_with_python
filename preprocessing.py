
import gensim
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import spacy
import warnings
from nltk.corpus import stopwords
from nltk.stem.porter import *

np.random.seed(1)

warnings.filterwarnings("ignore")
# Pulizia dati

stop_words = stopwords.words('english')
stop_words.extend(
    ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
     'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack',
     'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'would','still',
     'really', 'level', 'front', 'enough', 'round', 'around','oo_oo_oo_oo', 'none', 'never', 'tempo_switch','sometimes','bring',
     'everywhere', 'something', 'back','thing', 'give','well','little','know', 'when', 'else','mean', 'tell',
     'today'])

def Pulizia(text):

    def clean(testo):

        if (not pd.isnull(testo)) & (not pd.isna(testo)):

            # rimuovi testo come [Verse 1]
            lines = re.sub(r'\[.*\]', '', testo)

            #rimuovi i numeri
            lines = re.sub(r'\d+', ' ', lines)

            # rimuovi punteggiatura
            lines = re.sub(r'\W', ' ', lines)

            # fai lo split nel '\n'
            lines = [line.lower() for line in lines.split('\n') if len(line) > 0]

            # rimuovi il doppio spazio con un solo spazio
            lines = [re.sub(r'\s+', ' ', line) for line in lines]

            if lines:
                return lines

        return np.nan

    first_clean = text.apply(clean)

    def sent_to_words(sentences):
        for sent in sentences:
            # sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            # sent = re.sub('\s+', ' ', sent)  # remove newline chars
            # sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)

            yield sent

    def process_words(texts,
                      stop_words=stop_words,
                      allowed_postags=['NOUN', 'VERB', 'ADV'],
                      bigram_mod=None,
                      trigram_mod=None):


        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []

        # libreria spacy utile per lemmatizzazione e postag

        nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
        nlp.max_length = 10000000

        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

        # rimuoviamo le stop words dopo la lemmatizzazione
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word) > 3] for doc in texts_out]
        return texts_out


    col_one_list = first_clean.tolist()

    data_words = list(sent_to_words(col_one_list))

    # Costruzione bigrammi e trigrammi
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data = process_words(data_words,
                         bigram_mod=bigram_mod,
                         trigram_mod=trigram_mod)

    return data



























