
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


def Soup(url):
    session = requests.Session()
    session.headers[
        "User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    html = session.get(url).content
    return BeautifulSoup(html, "html.parser")


def Views(canzone="", artista=""):
    keyword = str(artista).replace("Featuring", "").split()[0] + "-" + str(canzone)
    url = "https://google.com/search?q=" + keyword + "(official video)"
    soup = Soup(url)
    l = soup.find_all("div", attrs={"class": "twQ0Be"})
    l = str(l)
    n = l.find("href")
    link = l[n:].split('"')[1]

    soup_y = Soup(link)

    between_script_tags = re.search('viewCount(.*)"', str(soup_y))

    stringa = str(between_script_tags)[str(between_script_tags).find('viewCount\\\\":'):].replace("\\", "")

    n_views = stringa.split('"')[2]

    return int(n_views)

def visual(df):
    visual = []
    for index, row in df.iterrows():

        try:
            t = Views(row['Titolo'], row['Artista'])
            print(row['Titolo'], "-", row['Artista'], "\n", "Visualizzazioni: ", t)
            visual.append(t)
            print("Done")
        except:
            t = "Errore"
            visual.append(t)
            print("Errore")

    df["YouTube"] = visual
    return df


if __name__ == '__main__':
    # PROVA
    # df[0:100]

    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

    df = visual(df)

    #df.to_csv("dataset/Dataset.csv", index=False, encoding='utf-8')