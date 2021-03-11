
from config import key_ibm
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class Getfeatures(object):

    def __init__(self):

        pass

    ## Prova per inserire delle feature alle pipeline del modello

    def analyze_tone(self, doc):

        # chiave nel file config
        authenticator = IAMAuthenticator(key_ibm)
        tone_analyzer = ToneAnalyzerV3(
            version='2017-09-21',
            authenticator=authenticator
        )

        tone_analyzer.set_service_url('https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/8d0a6e32-17f1-4b86-bb8b-b88655b43c0d')
        tone_analyzer.set_disable_ssl_verification(True)

        tones_output = tone_analyzer.tone(tone_input=doc,
                                          content_type="text/plain").get_result()

        return [i['tone_name'] for i in tones_output['document_tone']['tones']]

    def get_list_sentiment(self,df):

        column = df['Testo_stringa']
        first_clean = column.to_list()
        lista = []
        for i in first_clean:
            try:
                sentiment = self.analyze_tone(i)
                lista.append(sentiment)
                print(sentiment)
                print('Done')

            except:
                sentiment = 'Neutral'
                lista.append(sentiment)
                print('Neutral')
                print(lista)

        return lista

    def get_emotion(self,doc):
        return self.analyze_tone(doc)[0]


def get_custom_vectorizer(doc):
    return TfidfVectorizer(doc,ngram_range=(1, 2))

if __name__ == '__main__':

    df = pd.read_csv('dataset/Dataset.csv', error_bad_lines=False, sep=',')
    df2 = df.filter(['Titolo', 'Artista'], axis=1)

    ####### PROVA ######
    #df2 = df2[0:100]

    sent = Getfeatures()
    df2['Sentiment'] = sent.get_list_sentiment(df)

    df2.to_csv('dataset/Dataset.csv',index=False, encoding='utf-8')




