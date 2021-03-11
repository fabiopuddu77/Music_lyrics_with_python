
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(42)
import pandas as pd

def grafico_count(x, title):

    fig, ax = plt.subplots(nrows=1)
    freq = pd.crosstab(df[x], df["Dominant_Topic"])
    freq.plot(kind="bar", ax=ax)

    # , palette = ('#3F88B0', '#E8933B', '#A588C0',
    #              '#48A04A', '#CE534F', '#976F66',
    #              '#919191', '#DF9AC9')

    ax.set_title("Countplot " + title)
    ax.legend(title="Topics", loc=6, bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.8, hspace=0.6)
    plt.xticks(rotation=0)

    return plt.show()

def grafico_freq(x, title):

    fig, ax2 = plt.subplots(nrows=1)
    freq = pd.crosstab(df[x], df["Dominant_Topic"])
    relative = freq.div(freq.sum(axis=1), axis=0)
    relative.plot(kind="bar", ax=ax2)

    ax2.set_title("Relative frequency " + title)
    ax2.legend(title="Topics", loc=6, bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.8, hspace=0.6)
    plt.xticks(rotation=0)

    return plt.show()

'''Grafici genere anno 3 Topics'''

# df = pd.read_csv("dataset/df_topic_3.csv", error_bad_lines=False, sep=',')
# grafico_count(x='Genere', title="3 Topics")
# grafico_freq(x='Genere', title="3 Topics")
# grafico_count(x='Anno', title="3 Topics by Year")
# grafico_freq(x='Anno', title="3 Topics by Year")

'''Grafici per anno 5 Topics'''

df = pd.read_csv("dataset/df_topic.csv", error_bad_lines=False, sep=',')
grafico_count(x='Genere', title="5 Topics")
grafico_freq(x='Genere', title="5 Topics")

'''Decommentare il genere che si vuole visualizzare'''

genere = 'Country'
#genere = 'Pop'
#genere = 'Rock'
#genere = 'Blues'
#genere = 'R&B/Hip-Hop'
df = df.loc[(df['Genere'] == genere)]
grafico_count(x='Anno', title="{} Topics by Year".format(genere))
grafico_freq(x='Anno', title="{} by Year".format(genere))

'''Grafici sentiment 3 Topics'''
# df = pd.read_csv("dataset/df_topic_3.csv", error_bad_lines=False, sep=',')
# grafico_count(x='Sentiment', title='3 Topic by sentiment')
# grafico_freq(x='Sentiment', title='3 Topic by sentiment')

'''Grafici sentiment 5 Topics'''
# df = pd.read_csv("dataset/df_topic.csv", error_bad_lines=False, sep=',')
# grafico_count(x='Sentiment', title='5 Topic by sentiment')
# grafico_freq(x='Sentiment', title='5 Topic by sentiment')





