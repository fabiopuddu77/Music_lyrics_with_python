from textblob import TextBlob
from sentiment_analysis_spanish import sentiment_analysis
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import nltk
from nltk.corpus import wordnet as wn


sen_es = sentiment_analysis.SentimentAnalysisSpanish()


def remove_stop_words(tokens, language='english'):
    """
    Removes stop words and punctuation from tokens list
    :param language:
    :param tokens:
    :return:
    """
    if language != "english":
        punctuation = string.punctuation.replace("'", "")
    else:
        punctuation = string.punctuation

    return [w for w in tokens
            if w not in nltk.corpus.stopwords.words(language)
            and w not in punctuation]

def remove_non_english_words(tokens):
    """
    Removes all tokens without synsets
    :param tokens:
    :return:
    """
    return [w for w in tokens if wn.synsets(w)]

def tokens(lang, text, rmv_no_eng=False):
    if not rmv_no_eng:
        return remove_stop_words(language=lang, tokens=[w.lower() for w in text.split()])
    return remove_non_english_words(remove_stop_words(language=lang,
                                                      tokens=[w.lower() for w in text.split()]))

def most_used(lang, text, max_feature=100):
    fdist = FreqDist(tokens(lang, text, rmv_no_eng=False))
    return fdist.most_common(max_feature)

def get_sentiment(max_feature=100, testo="", lang='en'):
    if lang == "it":
        sentiment = 0
        for token in [t[0] for t in most_used("italian", testo, max_feature=100)]:
            sentiment += calculate_sentiment_it(token)
        return sentiment / max_feature * 100
    elif lang == "en":
        text = " ".join(tokens("english", testo, rmv_no_eng=True))
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    elif lang == "es":
        text = " ".join(tokens("spanish", testo, rmv_no_eng=False))
        p = sen_es.sentiment(text=testo)

        return (p*2)-1

def calculate_sentiment_it(token):
    lemma_sentiment = 0

    lexicon = pd.read_csv("dataset/sentix.csv",
                          sep="\t",
                          encoding='latin1')
    for row in lexicon[lexicon['lemma'] == token].values:
        lemma_sentiment += float(row[4]) - float(row[3])
    return lemma_sentiment


def show_graphics(dataset, SW=False):

    df = dataset

    if SW:

        ds = df.groupby(['Genere'])['sent_pol'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(20, 8))
        sns.barplot(x=ds['mean'],
                    y=ds.Genere,
                    palette="Dark2_r"
                    )
        plt.tight_layout()
        plt.show()

        ds = df.groupby(['Anno'])['sent_pol'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(20, 8))
        sns.barplot(x=ds['mean'],
                    y=ds.Anno,
                    palette="Dark2_r"
                    )
        plt.tight_layout()
        plt.show()

        ds = df.loc[df["Lingua"]=="en"].groupby(['Genere'])['sent_pol'].agg(['mean', 'std']).reset_index()
        print(ds)

        plt.figure(figsize=(20, 8))
        sns.barplot(y=ds['mean'],
                    x=ds.Genere,
                    palette="Dark2_r"
                    )
        plt.tight_layout()
        plt.show()

    #### Sentiment Analysis
if __name__ == '__main__':

    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

    ###### PROVA ######
    dataset = df[0:100]

    for index, row in dataset.iterrows():
        dataset.loc[index, 'sent_pol'] = get_sentiment(testo=row['Testo'], lang=row["Lingua"])

    #dataset.to_csv("dataset/Dataset_con_pol.csv")

    show_graphics(dataset, SW=True) # per visualizzare i grafici SW = T