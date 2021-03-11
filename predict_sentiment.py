
import pandas as pd
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def train_emotions(train, test, input):
    x_train = train['Testo_stringa']
    y_train = train['Sentiment']

    x_test = test['Testo_stringa']
    y_test = test['Sentiment']

    if input == "MNB":
        print("Multinomial Naive Bayes Classifier")
        mnb_model = Pipeline(
            steps=[
                ("combined_features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", MultinomialNB()),
            ]
        )
        mnb_model.fit(x_train, y_train)
        y_pred = mnb_model.predict(x_test)
        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for multinomial naive bayes: %s" % mnb_model.score(x_test, y_test))

        cm = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix', cm)
        conf_matr(input, cm, y_test, y_pred)

    if input == "LR":
        print("Logistic Regression Classifier")
        lr_model = Pipeline(
            steps=[
                ("features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(solver="liblinear", multi_class="ovr")),
            ]
        )
        lr_model.fit(x_train, y_train)
        y_pred = lr_model.predict(x_test)

        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for LogisticRegression: %s" % (lr_model.score(x_test, y_test)))

        cm = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix', cm)
        conf_matr(input, cm, y_test, y_pred)

    if input == 'DT':
        print("Decision Tree Classifier")
        dt_model = Pipeline(
            steps=[
                ("features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", DecisionTreeClassifier(max_depth=2)),
            ])
        dt_model.fit(x_train, y_train)
        y_pred = dt_model.predict(x_test)

        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for Decision Tree %s" % (dt_model.score(x_test, y_test)))
        cm = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix', cm)
        conf_matr(input, cm, y_test, y_pred)

    if input == 'SVC':
        print("Support Vector Classifier")
        svc_model = Pipeline(
            steps=[
                ("features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", SVC(kernel='linear', C=1)),
            ])
        svc_model.fit(x_train, y_train)
        y_pred = svc_model.predict(x_test)

        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for Support Vector Classifier %s" % (svc_model.score(x_test, y_test)))

        cm = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix', cm)
        conf_matr(input, cm, y_test, y_pred)

    if input == 'KNN':
        print("K-Neighbors Classifier")
        knn_model = Pipeline(
            steps=[
                ("features", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ])
        knn_model.fit(x_train, y_train)
        y_pred = knn_model.predict(x_test)

        print("Classification report: %s" % (classification_report(y_test, y_pred)))
        print("accuracy for K-Neighbors Classifier %s" % (knn_model.score(x_test, y_test)))

        cm = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix', cm)
        conf_matr(input, cm, y_test, y_pred)

    return


def train_test(df):
    train = df.sample(frac=0.7, random_state=123)
    test = df.drop(train.index)
    lista = test['Sentiment'].to_list()
    numero_sentiment = (nltk.FreqDist(lista).most_common())
    print('Sentiment',numero_sentiment)
    return train, test


def conf_matr(input, cm, y_test, y_pred):
    cm_df = pd.DataFrame(cm,
                         index=['Sadness','Joy'],
                         columns=['Sadness','Joy'])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True, fmt='d')

    plt.title('%s  Accuracy:%8.2f' % (input, accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.show()

def train_test(df):
    train = df.sample(frac=0.7, random_state=123)
    test = df.drop(train.index)
    return train, test

if __name__ == "__main__":

    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')
    df = df.loc[(df['Sentiment'] == 'Sadness') | (df['Sentiment'] == 'Joy')]
    train_df, test_df = train_test(df)

    '''SELEZIONARE IL MOELLO CHE SI VUOLE VISUALIZZARE MNB LR DT SVC KNN'''
    train_emotions(train_df, test_df, input='KNN')












