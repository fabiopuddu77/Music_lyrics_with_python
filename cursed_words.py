import pandas as pd
from argomenti import *
from better_profanity import profanity

def countWords0(Text):

    return 0

def countWords(Text, list_words):

    wordCount = 0
    WordsCounter = dict(zip(list_words, [0] * len(list_words)))

    for word in str(Text).split(" "):
        if word in list_words:
            wordCount += 1
            WordsCounter[word] += 1
    return wordCount


def numb_swear_words(text):
    if type(text) != str:
        text = str(text)
    else:
        pass

    text = text.split(" ")

    count: int = 0
    for t in text:
        if profanity.contains_profanity(t):
            count += 1

    return count

if __name__ == '__main__':

    df = pd.read_csv("dataset/Dataset.csv")

    ##### PROVA #####
    # df = df[0:100]

    lista_args = [('drinking',drinkingKeywords), ('love',loveKeywords), ('violence',violence),
                  ('drug',drug_smoking), ('sex',sex), ('money',money_power), ('car',car)]

    df['other'] = df.Testo.apply(countWords0)

    for index, row in df.iterrows():
        for l in lista_args:
            df.loc[index, l[0]] = countWords(Text=row['Testo'], list_words=l[1])

    df['volgare'] = df.Testo.apply(numb_swear_words)
    df["Topic"] = df[['other', 'drinking', "love", "violence", "drug", "sex", 'money', 'car', 'volgare']].idxmax(axis=1)

    #df.to_csv("Argomenti_forzati.csv")