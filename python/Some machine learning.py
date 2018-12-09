#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import re
import unicodedata
import pymystem3


# In[3]:


f = open('stop_words.txt', 'r')
stop_words = []
for line in f:
    stop_words.append(line.strip())
print(len(stop_words))
print(stop_words)


# In[4]:


#f = open('verses.txt', 'r')
f = open('zaum.txt', 'r')
verses = []
preprocessed_verses = []
verse = ''
mystem = pymystem3.Mystem()
words_count = 0

for line in f:
    if line.startswith(u'*'):
        if verse != '':
            preprocessed_verse = []
            for word in mystem.analyze(verse):
                try:
                    lemma = word['analysis'][0]['lex']
                    #if not lemma in stop_words:
                    preprocessed_verse.append(lemma)
                    print(lemma)
                except:
                    continue
            if preprocessed_verse != []:
                verses.append(verse)
                preprocessed_verses.append(preprocessed_verse)
            words_count += len(preprocessed_verse)
            verse = ''
    else:
        verse += line

f = open('verses.txt', 'r')
for line in f:
    if line.startswith(u'*'):
        if verse != '':
            preprocessed_verse = []
            for word in mystem.analyze(verse):
                try:
                    lemma = word['analysis'][0]['lex']
                    if not lemma in stop_words:
                        preprocessed_verse.append(lemma)
                        print(lemma)
                except:
                    continue
            if preprocessed_verse != []:
                verses.append(verse)
                preprocessed_verses.append(preprocessed_verse)
            words_count += len(preprocessed_verse)
            verse = ''
    else:
        verse += line
print('Words count: ' + str(words_count))


# In[5]:


model = gensim.models.Word2Vec(preprocessed_verses, size=150, window=15, min_count=1)
model.train(preprocessed_verses, total_examples=len(verses), epochs=15)


# In[6]:


len(model.wv.vocab)


# In[7]:


model.wv.vocab


# In[8]:


model.wv.most_similar('жена')


# In[9]:


def search(word, model, texts, preprocessed_texts):
    mystem = pymystem3.Mystem()
    word = mystem.lemmatize(word)[0]
    print(word)
    searched_texts = set()
    for i in range(len(preprocessed_texts)):
        if word in preprocessed_texts[i]:
            searched_texts.add(texts[i])
    for result in model.wv.most_similar(word):
        similar_word = result[0]
        for i in range(len(preprocessed_texts)):
            if similar_word in preprocessed_texts[i]:
                searched_texts.add(texts[i])
    return list(searched_texts)


# In[10]:


for text in search('любви', model, verses, preprocessed_verses):
    print(text)
    print('***')

