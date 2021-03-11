import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from config import client_access_token
import lyricsgenius as genius

geniusAPI = genius.Genius(client_access_token)


######### PROVA #######
classifiche = [{'name': 'Jazz', 'urlTag': 'jazz-digital-song-sales'}]

# classifiche = [{'name': 'Rock', 'urlTag': 'hot-rock-songs'},
#                {'name': 'Country', 'urlTag': 'hot-country-songs'},
#                {'name': 'R&B/Hip-Hop', 'urlTag': 'hot-r-and-and-b-hip-hop-songs'},
#                {'name': 'Dance/Electronic', 'urlTag': 'dance-electronic-streaming-songs'},
#                {'name': 'Pop', 'urlTag': 'pop-songs'},
#                {'name': 'Classical', 'urlTag': 'classical-digital-song-sales'},
#                {'name': 'Latin', 'urlTag': 'hot-latin-songs'},
#                {'name': 'Rap', 'urlTag': 'hot-rap-songs'},
#                {'name': 'Reggae', 'urlTag': 'reggae-digital-songs'},
#                {'name': 'Blues', 'urlTag': 'blues-digital-song-sales'},
#                {'name': 'Jazz', 'urlTag': 'jazz-digital-song-sales'}]
#

anni = [i for i in range(2002, 2019 + 1)]


def Billboard(anno):

    def diz_canzoni(url):

        for anno in anni:

            for classifica in classifiche:

                ''' Estrazione testo html'''
                page = requests.get(url, headers={'User-agent': 'your bot 0.1'}).text
                soup = bs(page, "html.parser")
                lista_class = soup.find_all("div", attrs={"class": "ye-chart-item__primary-row"})

                ''' Ricerca posizione, titolo e artista all'interno del testo html e posti all'interno
                di una lista di dizionari. Pulito successivamente per cercare i testi utilizzando i titoli
                su Genius'''
                classifica = []
                for i in lista_class:
                    classifica.append({'Rank': int(i.find("div", attrs={"class": "ye-chart-item__rank"}).text),
                                       'Titolo': i.find("div", attrs={"class": "ye-chart-item__title"}).text.strip(),

                                       'Artista': i.find("div",
                                                         attrs={"class": "ye-chart-item__artist"}).text.strip() \
                                      .replace(' x ', ' & ').replace(' X ', ' & ')})

        return classifica

    def genius_lyrics(Titolo, Artista):
        ''' Ricerca testo canzone col check in caso di testo non trovato'''
        try:
            return geniusAPI.search_song(Titolo, Artista).lyrics
        except:
            return

    def ricerca(classifiche):

        for classifica in classifiche:
            i = 0
            for canzone in classifica['liste']:
                i += 1

                # Visualizza a schermo la canzone che sto cercando
                print('Anno: ', anno)
                print('Genere: ', classifica['name'])
                print('Getting song', i, ':', canzone['Titolo'])

                canzone['Testo'] = genius_lyrics(canzone['Titolo'], canzone['Artista'])

                artistSplits = ['Featuring', 'With', 'And', '&', '/', ',']
                for splitter in artistSplits:

                    if canzone['Testo']:
                        break

                    canzone['Testo'] = genius_lyrics(canzone['Titolo'],
                                                     canzone['Artista'].split(splitter)[0].strip())

                if not canzone['Testo']:
                    canzone['Testo'] = genius_lyrics(canzone['Titolo'].split('(')[0].strip(),
                                                     canzone['Artista'])
        return

    def lista_DF(classifiche):

        classifiche_DF = None
        for classifica in classifiche:
            tempDf = pd.DataFrame.from_dict(classifica['liste'])
            tempDf['Genere'] = classifica['name']

            classifiche_DF = pd.concat([classifiche_DF, tempDf])

        classifiche_DF['Anno'] = anno

        classifiche_DF.reset_index(inplace=True, drop=True)

        classifiche_DF = classifiche_DF[['Anno',
                                         'Genere',
                                         'Rank',
                                         'Titolo',
                                         'Artista',
                                         'Testo']]

        print(classifiche_DF)

        return classifiche_DF

    for classifica in classifiche:
        classifica['url'] = ("https://www.billboard.com/charts/year-end" +
                             "/" + str(anno) + '/' + classifica['urlTag'])

    for classifica in classifiche:
        classifica['liste'] = diz_canzoni(classifica['url'])

    ricerca(classifiche)

    return lista_DF(classifiche)


def scaricare():
    class_totali_DF = None
    for anno in anni:
        classifiche_DF = Billboard(anno)
        class_totali_DF = pd.concat([class_totali_DF, classifiche_DF])

    canzoni_trovate = class_totali_DF.groupby(['Anno', 'Genere']).Genere.count()
    canzoni_trovate.name = 'Canzoni per Genere'

    testi_trovati = class_totali_DF.groupby(['Anno', 'Genere']).Testo.count()
    testi_trovati.name = 'Testi trovati'

    perc_testi_trovati = (class_totali_DF.groupby(['Anno', 'Genere']).Testo.count() /
                          class_totali_DF.groupby(['Anno', 'Genere']).Genere.count())
    perc_testi_trovati.name = 'songsFoundPercentage'

    Summary = pd.concat([canzoni_trovate, testi_trovati, testi_trovati], axis=1)

    print(Summary)
    print('\n================\n')
    print('Canzoni Totali:', len(class_totali_DF))
    print('Canzoni Trovate:', class_totali_DF.Testo.count())
    print('Canzoni non trovate:', len(class_totali_DF) - class_totali_DF.Testo.count())
    print('Percentuale di canzoni trovate',
          class_totali_DF.Testo.count() * 100 / len(class_totali_DF))

    #class_totali_DF.to_csv('Dataset.csv',index=False, encoding='utf-8')

    return class_totali_DF

if __name__ == '__main__':
    df = scaricare()
    #df.to_csv('dataset/Dataset.csv')