# TF-IDF

import pandas as pd
import numpy as np

dataset = pd.read_csv('clean_dataset_stem.csv', sep=';')
dataset_feature = dataset['ProcessedText'].astype(str)

print(dataset_feature)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset_feature)

# print(X.shape)

features = vectorizer.get_feature_names_out()
# print(features)

idfValues = vectorizer.idf_
d = dict(zip(features, 9 - idfValues))
sortedDict = sorted(d.items(), key = lambda d: d[1], reverse=True)

for i in range(3867):
        print(sortedDict[i])


from wordcloud import WordCloud
import matplotlib.pyplot as plot

def PlotWordCloud(frequency):
    worcloudPlot = WordCloud(background_color="white", width=1500, height=1000)
    worcloudPlot.generate_from_frequencies(frequncies=frequency)
    wordcloud = WordCloud().generate(dataset)
    plot.figure(figsize=(15,10))
    plot.imshow(worcloudPlot, interpolation="bilinear")
    plot.axis("off")
    plot.show()
    PlotWordCloud(d)