
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

def grafico1(df):

    ax = sns.countplot(x="Anno", hue="Sentiment", data=df,  palette=('#3F88B0','#E8933B','#A588C0',
                                                                     '#48A04A','#CE534F','#976F66',
                                                                     '#919191','#DF9AC9'))

    ax.set(xlabel='', ylabel='', title='Sentiment by Year')

    ax.legend().set_title('')

    return plt.show()

def grafico2(genere, ax):

    ax.set(xlabel='', ylabel='', title='Sentiment {} by Year'.format(genere))
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))

    ax.legend().set_title('')

    return plt.show()

blu = '#3F88B0'
giallo = '#E8933B'
viola= '#A588C0'
verde = '#48A04A'
rosso = '#CE534F'
marron = '#976F66'
grigio = '#919191'
rosa = '#DF9AC9'

palette=([giallo,blu,rosa,rosso, marron,grigio,verde, viola])

def grafico_freq(x, dataset, title):

    fig, ax2 = plt.subplots(nrows=1)
    freq = pd.crosstab(dataset[x], dataset["Sentiment"])
    relative = freq.div(freq.sum(axis=1), axis=0)
    relative.plot(kind="bar", ax=ax2)
    ax2.set_title("Relative frequency " + title)
    ax2.legend(title="Sentiment", loc=6, bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.8, hspace=0.6)
    plt.xticks(rotation=0)



    return plt.show()

'''Grafici Countplot'''

grafico1(df=df)
genere = 'Country'
df2 = df.loc[df['Genere'] == genere]
grafico2(genere, ax = sns.countplot(x="Anno", hue="Sentiment", data=df2, palette=(giallo,blu,rosa,rosso, marron,grigio,
                                                                                  verde, viola)))

genere = 'Rock'
df2 = df.loc[df['Genere'] == genere]
grafico2(genere,ax = sns.countplot(x="Anno", hue="Sentiment", data=df2, palette=(blu,giallo,viola,verde,rosso, marron,
                                                                                 grigio, rosa)))

genere = 'R&B/Hip-Hop'
df2 = df.loc[df['Genere'] == genere]
grafico2(genere,ax = sns.countplot(x="Anno", hue="Sentiment", data=df2, palette=(grigio, rosso, giallo, marron, verde,
                                                                                 viola, blu, rosa)))

'''Grafici frequenza'''

anno = 'Anno'
grafico_freq(x='Anno', dataset = df, title="by year")


genere = 'Country'
df2 = df.loc[df['Genere'] == genere]
grafico_freq(x='Anno', dataset = df2, title="{} by Year".format(genere))

genere = 'R&B/Hip-Hop'
df2 = df.loc[df['Genere'] == genere]
grafico_freq(x='Anno', dataset = df2, title="{} by Year".format(genere))

genere = 'Rock'
df2 = df.loc[df['Genere'] == genere]
grafico_freq(x='Anno', dataset = df2, title="{} by Year".format(genere))