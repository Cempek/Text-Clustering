# Importing the libraries
import os 
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
pm = PorterStemmer()
from nltk.probability import FreqDist

# Reading the csv file from the link

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
import requests
import csv
from contextlib import closing
data = []
with closing(requests.get(url)) as f:
    csv_file = (line.decode('utf-8') for line in f.iter_lines())
    reader = csv.reader(csv_file , delimiter=',', quotechar='"')
    for row in reader:
        data.append(row)

# Clean the text 
# 2 different stemming methods have been used to obtain cleaned words

data = data[1:]

bbc_array = np.asarray(data)
categories = np.unique(bbc_array[:,0])

temp_text = []
cleaned_texts = []

for i in range(len(data)):
    temp_text.append([])
    temp_text[i] = re.sub('[^a-zA-Z]', ' ', data[i][1] )
    temp_text[i] = temp_text[i].lower()
    temp_text[i] = temp_text[i].split()
 #   temp_text[i] = [lm.lemmatize(word) for word in temp_text[i] if not word in set(stopwords.words('english')) ]           # First stemming method 
    temp_text[i] = [pm.stem(word) for word in temp_text[i] if not word in set(stopwords.words('english'))]                  # Second stemming method
 #   temp_text[i] = [word for word in temp_text[i] if not word in set(stopwords.words('english')) ]                         # we didn't use stemming method, just get rid of stopwords

    cleaned_texts.append(temp_text[i])
   
# Vectorize all words

from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer

model = Word2Vec(cleaned_texts, min_count = 1)
X = list(model.wv.vocab)

def word_sentinizer(txt, model):
    text_vect = []
    no_words = 0
    for word in txt:
        if no_words ==  0:
            text_vect = model[word]
        else:
            text_vect = np.add(text_vect, model[word])
        no_words += 1
    return np.asarray(text_vect) / no_words

X = []
for text in cleaned_texts:
    X.append(word_sentinizer(text, model))
    
no_cluster = len(categories)

# Clustering vectorized words
kclusterer = KMeansClusterer(no_cluster, distance= nltk.cluster.util.cosine_distance, repeats = 100)
assigned_clusterers = kclusterer.cluster(X, assign_clusters = True)

print(assigned_clusterers)


# Stacking output and predicted results

cluster_results = np.asarray(assigned_clusterers) 
cluster_results = cluster_results.reshape(len(cluster_results), 1)
cleaned_texts = np.asarray(cleaned_texts)
cleaned_texts = cleaned_texts.reshape(len(cleaned_texts), 1)

results = np.hstack((cleaned_texts,cluster_results, bbc_array[:,0].reshape(len(bbc_array), 1)))


# Combining the texts based on thier clusters to find clusters' topics. So we will 5 long texts
# We combined the text by clusters because we need to discover which cluster belogns to which topic 
  
text_by_clusters = []
for i in range(no_cluster):
    text_by_clusters.append([[],[]])
    
    
for i in range(no_cluster):
    for k in range(len(results)):
        if results[k,1] == i:
            temp = " ".join(results[k,0])
            text_by_clusters[i][0].append(str(temp))
            text_by_clusters[i][1] = i

for i in range(no_cluster):
    text_by_clusters[i][0] = " ".join(text_by_clusters[i][0])

# First way to find the clusters' topic. ---> Creating word cloud for each cluster

from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,5, figsize = (25,5))
for i in range(len(text_by_clusters)):
    wordcloud = WordCloud(background_color = 'white',
                              width = 1200,
                              height = 1200).generate(text_by_clusters[i][0]) 
    ax[i].imshow(wordcloud)
    ax[i].grid(False)
    ax[i].axis('off')
    ax[i].title.set_text(str(text_by_clusters[i][1]))

# Second way to find the clusters' topic. ---> Finding most common 20 words and print them based on their cluster
topic = [[],[],[]]

for i, text in enumerate(text_by_clusters):
    tokenized_words = nltk.tokenize.word_tokenize(text[0])
    word_dist = FreqDist(tokenized_words)
    for word, frequency in word_dist.most_common(20):
        topic[0].append(int(text[1]))
        topic[1].append(word)
        topic[2].append(frequency)

topic = np.array(topic).T
topic= pd.DataFrame(topic)
topic[0] = topic[0].astype(int)

for i in range(len(np.unique(topic.iloc[:,0]))):
    common_words = topic[topic.iloc[:,0] == i].iloc[:,1]
    print(f'{i}. class most common words are {[common_words.iloc[a] for a in range(len(common_words))]}')

 
# Evaluating the results

evaluating = {'cluster' : [], 'no_record' : [], 'correct_pred' : [], 'wrong_pred' : [] }
clusters, counts = np.unique(results[:,2], return_counts = True)
clusters = np.asarray([clusters, counts]).T
for i in range(len(clusters)):
    evaluating['cluster'].append(clusters[i,0])
    evaluating['no_record'].append(clusters[i,1])
    evaluating['correct_pred'].append(0)
    evaluating['wrong_pred'].append(0)

evaluating = pd.DataFrame(evaluating)

false = 0

for i in range(len(results)):
    if ((results[i][1] == 1) and (results[i][2]== 'business')):
        evaluating.iloc[0,2] += 1      
    elif ((results[i][1] == 4) and (results[i][2]== 'sport')):
        evaluating.iloc[3,2] += 1
    elif ((results[i][1] == 3) and (results[i][2]== 'entertainment')):
        evaluating.iloc[1,2] += 1
    elif ((results[i][1] == 0) and (results[i][2]== 'tech')):
        evaluating.iloc[4,2] += 1
    elif ((results[i][1] == 2) and (results[i][2]== 'politics')):
        evaluating.iloc[2,2] += 1
    else:
        false +=1
    

evaluating.iloc[:,3] = evaluating.iloc[:,1] - evaluating.iloc[:,2]


print(f'{round((sum(evaluating.iloc[:,2])/len(results))*100,2)} of the news are predicted as correct')

for i in range(len(clusters)):
    print(f'{round((evaluating.iloc[i,2]/evaluating.iloc[i,1])*100,2)} of the news that is related to {evaluating.iloc[i,0]} is predicted correctly')
    
    