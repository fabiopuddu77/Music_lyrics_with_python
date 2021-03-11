
''' Nel main possiamo far partire il codice intero, se vogliamo si possono far partire gli script singolarmente
    dal loro interno. Sono stati lasciati i main al loro interno'''

import pandas as pd
from termcolor import colored
import string
from preprocessing import Pulizia
from preprocessing_es import Pulizia_es


if __name__ == '__main__':

    #### Estrazione dei dati

    SW = input("Se si vuole scaricare il dataset, inserire yes\n se si vuole leggere il dataset inserire no\n")
    try :
        if SW == "yes":
            from spider_scrapper import scaricare
            from analisi_testuale import language_detect
            from visualizzazioni_youtube import visual

            df = scaricare()

            df['Lingua'] = language_detect(df)

            df = visual(df)

            df.to_csv('dataset/Dataset.csv')
            df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

        elif SW == "no":
            df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')
            ##### PROVA ######
            #df = df [0:100]
        else:
            print("valore non valido ")
    except:
        print("valore non valido ")


    #### statistiche generali
    # from analisi_testuale import grafici1,grafici2
    #
    # print(colored(colored(df.head(), 'green')))
    #
    # print("Variabile Genere: \n\n", df.Genere.unique())
    #
    # print("dimensioni dataset: ", colored(colored(df.shape, 'green')))
    #
    # print("Variabile Genere: \n\n", colored(colored(df.Genere.describe(), 'green')))
    #
    # for index, row in df.iterrows():
    #     tok = row["Testo"].replace("\n", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
    #     df.loc[index, 'lunghezza'] = len(tok)
    #     df.loc[index, 'Lex'] = len(set(tok))
    #
    # grafici1(df)
    # grafici2(df)
    #
    # from cursed_words import *
    #
    # lista_args = [('drinking', drinkingKeywords), ('love', loveKeywords), ('violence', violence),
    #               ('drug', drug_smoking), ('sex', sex), ('money', money_power), ('car', car)]
    #
    # df['other'] = df.Testo.apply(countWords0)
    #
    # for index, row in df.iterrows():
    #     for l in lista_args:
    #         df.loc[index, l[0]] = countWords(Text=row['Testo'], list_words=l[1])
    #
    # df['volgare'] = df.Testo.apply(numb_swear_words)
    # df["Topic"] = df[['other', 'drinking', "love", "violence", "drug", "sex", 'money', 'car', 'volgare']].idxmax(axis=1)
    #
    #
    # # Topic Modeling
    # from coherence import *
    #
    # '''Coherence Dataset Totale '''
    # df = df.loc[(df['Lingua'] == 'en')]
    #
    # '''Coherence Latin e lingua spagnola'''
    # # df = df.loc[(df['Lingua'] == 'es') & (df['Genere'] == 'Latin')]
    # # data_classes = ['Latin']
    #
    # testo = df['Testo']
    # data_ready = Pulizia(testo)
    #
    # # Create Dictionary
    # id2word = corpora.Dictionary(data_ready)
    #
    # # Create Corpus: Term Document Frequency
    # corpus = [id2word.doc2bow(text) for text in data_ready]
    #
    # model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready,
    #                                                         start=2, limit=8, step=1)
    #
    # # Show graph
    # limit = 8
    # graph_coherence(coherence_values, limit, start=2, step=1)

    from topic_modelling_genre import *

    '''VEDREMO UNA LISTA DI TOPIC MODELLING DI DIVERSI DATASET, DECOMMENTARE IL DATASET CHE SI VUOLE ANALIZZARE'''

    '''  1.    Per la topic modelling con 3 Topics trovati con la coherence decommentare il codice'''
    data_classes = ['Totale_3_']
    n_topics = 3

    '''  2.    Per la migliore suddivisione con 5 Topics decommentare il codice'''
    # data_classes = ['Totale_5_']
    # n_topics = 5

    '''   3.    Topic Analisys Approfondimento, con il dataset avente i 3 generi più frequenti con la lingua inglese

     Se si vorrà visualizzare la il colore dei cluster con il genere sostituire a colors = topic_num
          la variabile "genere" nel plot.scatter che diventerà colors = genere,
           
          selezioniamo i tre generi più frequenti inglesi Country, Rock e R&B/Hip-Hop per vedere se si possono
          associare a tre topic distinti'''

    # df = df.loc[(df['Genere'] == 'Country') | (df['Genere'] == 'R&B/Hip-Hop') | (df['Genere'] == 'Rock')]
    # n_topics = 3
    # data_classes = ['Country','R&B/Hip-Hop','Rock']
    # genere = df_gen['Genere'].apply(data_classes.index).tolist()
    # testo_gen = df_gen['Testo']
    # data_ready = Pulizia(testo_gen)

    '''  4.     Topic Analisys Approfondimento, con il dataset avente il singolo genere si sono selezionati i 3 
    generi più frequenti nel dataset, deselezionare a seconda di quale topic analysis si vuole fare'''

    ''' Country'''
    # df = df.loc[(df['Genere'] == 'Country')]
    # data_classes = ['Country']
    # n_topics = 5

    '''Rock'''
    # df = df.loc[(df['Genere'] == 'Rock')]
    # data_classes = ['Rock']
    # n_topics = 4

    '''R&B HipHop'''
    # df = df.loc[(df['Genere'] == 'R&B/Hip-Hop')]
    # data_classes = ['R&BHipHop']
    # n_topics = 5


    data_ready = Pulizia(df.Testo)
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    compute_lda(corpus=corpus, id2word=id2word, n_topics=n_topics, data_ready=data_ready, nome=data_classes)

    '''Latin'''

    # df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')
    # df = df.loc[(df['Lingua'] == 'es') & (df['Genere'] == 'Latin')]
    #
    # ##### PROVA DF PIU PICCOLO ######
    # df = df[0:100]
    #
    # data_classes = ['Latin']
    # n_topics = 2
    # testo = df['Testo']
    # data_ready = Pulizia_es(testo)
    #
    # # Create Dictionary
    # id2word = corpora.Dictionary(data_ready)
    #
    # # Create Corpus: Term Document Frequency
    # corpus = [id2word.doc2bow(text) for text in data_ready]
    #
    # # Build LDA model
    #
    # compute_lda(corpus=corpus, id2word=id2word, n_topics=n_topics, data_ready=data_ready, nome=data_classes)

    # Sentiment
    # from sentiment_polarita import *
    #
    # # PROVA
    # dataset = df[0:100]
    #
    # for index, row in dataset.iterrows():
    #     dataset.loc[index, 'sent_pol'] = get_sentiment(testo=row['Testo'], lang=row["Lingua"])
    #
    # dataset.to_csv("dataset/Dataset_con_pol.csv")
    #
    # show_graphics(dataset, SW=True) # per visualizzare i grafici SW = T



    # Sentiment IBM
    from sentiment_ibm import *

    df2 = df.filter(['Titolo', 'Artista'], axis=1)

    ####### PROVA ######
    df2 = df2[0:100]

    sent = Getfeatures()
    df2['Sentiment'] = sent.get_list_sentiment(df)

    df2.to_csv('dataset/Dataset.csv', index=False, encoding='utf-8')

    # Predict Genre
    from predict_genre import *

    '''Se mettessimo tutto il dataset otteniamo un classificatore non performante, 
       si sono selezionati solo i generi più presenti nel ds'''

    '''Selezionare il modello che si vuole visualizzare nell'input MNB LR SVC KNN'''

    df = df.loc[(df['Genere'] == 'Rock') | (df['Genere'] == 'Country') | (df['Genere'] == 'R&B/Hip-Hop')
                | (df['Genere'] == 'Latin')]
    train_df, test_df = train_test(df)

    '''Decommentare il modello che si vuole visualizzare'''
    train_emotions(train_df, test_df, input='MNB')
    # train_emotions(train_df, test_df, input='LR')
    # train_emotions(train_df, test_df, input='SVC')
    # train_emotions(train_df, test_df, input='KNN')

    '''Decommentare per il modello Decisional Tree'''
    # df = df.loc[(df['Genere'] =='R&B/Hip-Hop')
    #             | (df['Genere'] == 'Latin')]
    # train_df, test_df = train_test(df)
    # train_emotions(train_df, test_df, input='DT')

    # Predict Sentiment
    from predict_sentiment import *

    df = df.loc[(df['Sentiment'] == 'Sadness') | (df['Sentiment'] == 'Joy')]
    train_df, test_df = train_test(df)

    '''SELEZIONARE IL MOELLO CHE SI VUOLE VISUALIZZARE MNB LR DT SVC KNN'''
    train_emotions(train_df, test_df, input='KNN')

