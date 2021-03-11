from bs4 import BeautifulSoup
import requests
import pandas as pd

df = pd.read_csv("dataset/Dataset.csv")
df = df[0:20]

artisti = [artista for artista in df.Artista]
titoli = [titolo for titolo in df.Titolo]


def informazioni( artista=""):
    keyword = str(artista).split()[0]
    if len(artista.split())>1:
        if artista.split()[1].lower() != "featuring":
            keyword = str(artista).split()[0] + "+" + str(artista).split()[1]
    url = "https://google.com/search?q=" + keyword

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    html = session.get(url).content
    soup = BeautifulSoup(html, "html.parser")

    info1 = [s.text for s in soup.find_all("span", attrs={"class": "w8qArf"})]

    info2 = [s.text for s in soup.find_all("span", attrs={"class": "LrzXr kno-fv"})]

    return [ {info1[i]:info2[i]} for i in range(len(info2))]


informazioni_cantante = []

for art in artisti:
    a = informazioni(art)
    print(art,a)
    informazioni_cantante.append(a)


df = df.assign( informazioni = informazioni_cantante)


for index,row in df.iterrows():
    newDict = {}
    for element in row['informazioni']:
        for k,v in element.items():
            newDict[k] = v
    df.at[index, 'informazioni']= newDict

df.to_csv('musica_con_informazioni.csv')