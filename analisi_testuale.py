
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
from langdetect import detect


def language_detect(text):
    if type(text) != str:
        text = str(text)
    else:
        pass
    if "[Instrumental]" in text:
        return "instrumental"
    if "nan" in text:
        return "nan"
    try:
        lang = detect(text)
        print(lang)
        return lang
    except:
        return "nan"



def grafici1(df):
    plt.figure(figsize=(20, 8))
    sns.boxplot(data=df,
                x="Genere",
                y="Lex",
                palette="Dark2_r",
                )
    plt.title('Lexical Diversity by Genre')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))
    sns.boxplot(data=df,
                x="Genere",
                y="lunghezza",
                palette="Dark2_r",
                )
    plt.title('Lunghezza by Genre')
    plt.tight_layout()
    plt.show()

def grafici2(df):

    ds = df.groupby(['Genere'])['lunghezza'].agg(['mean', 'std']).reset_index()
    print(ds)

    plt.figure(figsize=(20, 8))
    sns.barplot(y=ds['mean'],
                x=ds.Genere,
                palette="Dark2_r",

                )
    plt.title('Numero parole medio per testi by Genre')
    plt.tight_layout()
    plt.show()


    ds = df.groupby(['Genere'])['Lex'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(20, 8))
    sns.barplot(y=ds['mean'],
                x=ds.Genere,
                palette="Dark2_r"
                )
    plt.title('Media Lexical Diversity by Genre')
    plt.tight_layout()
    plt.show()

    ds = df.groupby(['Lingua'])['Lex'].agg(['mean', 'std']).reset_index()
    print(ds)

    plt.figure(figsize=(20, 8))
    sns.barplot(y=ds['mean'],
                x=ds.Lingua,
                palette="Dark2_r"
                )
    plt.title('Lexical Diversity by Language')
    plt.tight_layout()
    plt.show()


    ds = df.groupby(['Anno'])['Lex'].agg(['mean', 'std']).reset_index()
    print(ds)

    plt.figure(figsize=(20, 8))
    sns.barplot(y=ds['mean'],
                x=ds.Anno,
                palette="Dark2_r"
                )

    plt.title('Lexical Diversity by year')
    plt.tight_layout()
    plt.show()

    ds = df.groupby(['Anno'])['Lex'].agg(['mean', 'std']).reset_index()
    print(ds)

    plt.figure(figsize=(20, 8))
    sns.lineplot(y=ds['mean'],
                x=ds.Anno,
                palette="Dark2_r"
                )

    plt.title('Lineplot Lexical Diversity by year')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv("dataset/Dataset.csv")

    for index, row in df.iterrows():
        tok = row["Testo"].replace("\n", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
        df.loc[index, 'lunghezza'] = len(tok)
        df.loc[index, 'Lex'] = len(set(tok))

    grafici1(df)
    grafici2(df)