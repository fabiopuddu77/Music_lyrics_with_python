import warnings
from collections import Counter
from pprint import pprint
import gensim
import gensim.corpora as corpora
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from PIL import Image
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, save
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
from preprocessing import Pulizia

warnings.filterwarnings("ignore")


def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        if len(row_list) == 0:
            continue
        row = row_list[0] if ldamodel.per_word_topics else row_list
        if isinstance(row, tuple):
            row = [row]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def visualize_topics(lda_model, corpus, nome):
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, mds='mmds')
    pyLDAvis.save_html(vis, './report/topic_modeling_visualization_{}.html'.format(nome))


def show_topic_clusters_gen(lda_model, corpus, nome, n_topics):

    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc by genre

    # tSNE Dimension Reduction
    # t-distributed Stochastic Neighbor Embedding
    tsne_model = TSNE(perplexity=50, n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    topic_num = np.argmax(arr, axis=1)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    file_name = 'report/topic_modeling_clusters_{}.html'.format(nome)
    output_file(file_name)

    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

    plot = figure(title="Genre {} t-SNE Clustering of {} LDA Topics".format(nome,n_topics),
                  plot_width=900, plot_height=700)

    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    save(plot)

def plot_document(df):
    doc_lens = [len(d) for d in df.Text]

    # Plot
    plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(doc_lens, bins=1000, color='navy')
    plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750, 90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750, 80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750, 70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750, 60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 400), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=30)
    plt.xticks(np.linspace(0, 400, 9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    return plt.show()

def wordcloud_topic(lda_model):
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



    # stylecloud.gen_stylecloud(file_path='constitution.txt',
    #                           colors=['#ecf0f1', '#3498db', '#e74c3c'],
    #                           background_color='#1A1A1A')
    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1500,
                      height=1000,
                      max_words=50,
                      colormap='tab10', prefer_horizontal=0.9,
                      mask = np.array(Image.open('file_and_images/cerchio-nero.jpg')))


    topics = lda_model.show_topics(num_words=50, formatted=False)

    fig, axes = plt.subplots(1, n_topics, figsize=(12, 12), sharey=True, dpi=120)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=250)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=20))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    return plt.show()

def key_words(lda_model):
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, dpi=120)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.20);
        ax.set_ylim(0, 5500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=15)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left');
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=15, y=1.05)
    return plt.show()

from matplotlib.patches import Rectangle

def sentences_chart(lda_model, corpus, start = 0, end = 8):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(8, (end-start)*0.95), dpi=150)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1]
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ":      ",horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=10, color='black', transform=ax.transAxes, fontweight=500)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=1))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 9:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=10, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=300)
                    word_pos += .018 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=10, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=10, y=0.95, fontweight=400)
    plt.tight_layout()
    plt.show()


def compute_lda(corpus, id2word, n_topics, data_ready, nome=""):
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()

    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # df_dominant_topic.to_csv('df_dominant_topic.csv')

    # Mostrami a quale topic appartiene il documento numero 1
    df1 = df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Document_No
    print('Il documento numero: ', df1.to_string(index=False))

    df2 = df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Dominant_Topic
    print('Appartiene al topic ', df2.to_string(index=False))

    # plot_document(df_dominant_topic)
    # wordcloud_topic(lda_model)
    # key_words(lda_model)
    # sentences_chart(lda_model, corpus)

    # ''' Visualize HTML reports of topics and topic clusters by genre '''
    # visualize_topics(lda_model, corpus, nome=data_classes[0])
    show_topic_clusters_gen(lda_model, corpus, nome=nome, n_topics=n_topics)

    ''' Visualize HTML reports of topics and topic clusters 3 genre'''
    # visualize_topics(lda_model, corpus, nome='Rock_Country_R&BHH')
    # show_topic_clusters_gen(lda_model, corpus, nome='Rock_Country_R&BHH_gencolor', n_topics=n_topics)



if __name__ == '__main__':

    # lettura dataset
    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

    ''' Dataset Totale '''
    df = df.loc[(df['Lingua'] == 'en')]

    ######## PER PROVARE SE FUNZIONA UTILIZZIAMO UN DATASET PIù PICCOLO #####
    #df = df[0:100]

    '''VEDREMO UNA LISTA DI TOPIC MODELLING DI DIVERSI DATASET, DECOMMENTARE IL DATASET CHE SI VUOLE ANALIZZARE'''

    '''  1.    Per la topic modelling con 3 Topics trovati con la coherence decommentare il codice'''
    data_classes = ['Totale_3_']
    n_topics = 3
    testo = df['Testo']
    data_ready = Pulizia(testo)

    '''  2.    Per la migliore suddivisione con 5 Topics decommentare il codice'''
    # data_classes = ['Totale_5_']
    # n_topics = 5
    # testo = df['Testo']
    # data_ready = Pulizia(testo)

    '''   3.    Topic Analisys Approfondimento, con il dataset avente i 3 generi più frequenti con la lingua inglese
     
     Se si vorrà visualizzare la il colore dei cluster con il genere sostituire a colors = topic_num
          la variabile "genere" nel plot.scatter che diventerà colors = genere, 
          selezioniamo i tre generi più frequenti inglesi Country, Rock e R&B/Hip-Hop per vedere se si possono
          associare a tre topic distinti'''

    # df_gen = df.loc[(df['Genere'] == 'Country') | (df['Genere'] == 'R&B/Hip-Hop') | (df['Genere'] == 'Rock')]
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
    # testo = df['Testo']
    # n_topics = 5
    # data_ready = Pulizia(testo)

    '''Rock'''
    # df = df.loc[(df['Genere'] == 'Rock')]
    # testo = df['Testo']
    # data_classes = ['Rock']
    # n_topics = 4
    # data_ready = Pulizia(testo)

    '''R&B HipHop'''
    # df = df.loc[(df['Genere'] == 'R&B/Hip-Hop')]
    # data_classes = ['R&BHipHop']
    # n_topics = 5
    # testo = df['Testo']
    # data_ready = Pulizia(testo)


    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()

    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    '''SE SI VUOLE SALVARE IL DATASET CON I TOPIC APPARTENENTI A OGNI OSSERVAZIONE DECOMMENTARE'''
    #df_dominant_topic.to_csv('df_dominant_topic.csv')


    # Mostrami a quale topic appartiene il documento numero 1
    df1 =  df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Document_No
    print('Il documento numero: ',df1.to_string(index=False))

    df2 = df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Dominant_Topic
    print('Appartiene al topic ', df2.to_string(index=False))

    '''Decommentare il grafico che si vuole visualizzare'''
    #plot_document(df_dominant_topic)
    wordcloud_topic(lda_model)
    #key_words(lda_model)
    #sentences_chart(lda_model, corpus)


    ''' Visualize HTML reports of topics and topic clusters by genre '''
    visualize_topics(lda_model, corpus, nome=data_classes[0])
    show_topic_clusters_gen(lda_model, corpus, nome=data_classes[0], n_topics=n_topics)

    '''Grafici del dataset con i 3 generi più frequenti'''
    #visualize_topics(lda_model, corpus, nome='Rock_Country_R&BHH')
    #show_topic_clusters_gen(lda_model, corpus, nome='Rock_Country_R&BHH_gencolor', n_topics=n_topics)

