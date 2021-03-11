
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

def popolarita(x,df,title):


    df['YouTube'].replace('', np.nan, inplace=True)

    df['YouTube'].replace('errore', np.nan, inplace=True)

    df['YouTube'].replace('Errore', np.nan, inplace=True)

    df.dropna(subset=['YouTube'], inplace=True)

    df['YouTube'] = df.YouTube.astype(float)

    ax = sns.barplot(x='YouTube', y =x, data=df, palette="Blues_d")

    ax.set(xlabel='', ylabel='', title=title)

    ax.legend().set_title('')

    return plt.show()

def grafico_freq(x, df, title):
    df['YouTube'].replace('', np.nan, inplace=True)
    df['YouTube'].replace('errore', np.nan, inplace=True)
    df['YouTube'].replace('Errore', np.nan, inplace=True)
    df.dropna(subset=['YouTube'], inplace=True)
    df['YouTube'] = df.YouTube.astype(float)

    fig, ax2 = plt.subplots(nrows=1)
    freq = pd.crosstab(df[x], df["YouTube"])
    relative = freq.div(freq.sum(axis=1), axis=0)
    relative.plot(kind="bar", ax=ax2)
    ax2.set_title("Relative frequency " + title)
    ax2.legend(title="Popolarita", loc=6, bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.8, hspace=0.6)
    plt.xticks(rotation=0)
    return plt.show()

def grafico1(df,genere):

    ax = sns.countplot(x="Anno", data=df)

    ax.set(xlabel='', ylabel='', title='{} popularity by Year'.format(genere))

    ax.legend().set_title('')

    return plt.show()

popolarita(x='Sentiment', df=df, title='Sentiment by Popularity')
popolarita(x='Genere',df=df, title='Popularity by Genre')



