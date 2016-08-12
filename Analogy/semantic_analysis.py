from gensim import corpora, models, matutils, similarities
from pprint import pprint

import random



def generate_corpus(a1):

    tmpd = [(f.name,f.text) for f in a1.features.values()]

    #tmpd = [random.choice(tmpd) for i in range(500)]

    lookup = {}
    documents = []
    for i,(a,b) in enumerate(tmpd):
        lookup[i] = a
        documents.append(b)

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus, lookup


def get_similarity(a1, a, b, corpus, dictionary, lookup):
    #lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)

    #doc1 = a1.features[a].text
    #vec_bow1 = dictionary.doc2bow(doc1.lower().split())
    #vec_lsi1 = lsi[vec_bow1] # convert the query to LSI space

    #doc2 = a1.features[b].text
    #vec_bow2 = dictionary.doc2bow(doc2.lower().split())
    #vec_lsi2 = lsi[vec_bow2] # convert the query to LSI space

    #index = similarities.MatrixSimilarity(lsi[corpus])

    #sims = index[vec_lsi1]

    #nx = [(v,lookup[i]) for i,v in enumerate(sims)]

    #pprint(sorted(nx,reverse=True))
    #print(matutils.cossim(vec_lsi1, vec_lsi2))


    #lda = models.LdaModel(corpus,id2word=dictionary,passes=1)

    #doc1 = a1.features[a].text
    #vec_bow1 = dictionary.doc2bow(doc1.lower().split())
    #vec_lda1 = lda[vec_bow1] # convert the query to LDA space

    #doc2 = a1.features[b].text
    #vec_bow2 = dictionary.doc2bow(doc2.lower().split())
    #vec_lda2 = lda[vec_bow2] # convert the query to LDA space

    #index = similarities.MatrixSimilarity(lda[corpus])

    #sims = index[vec_lda1]

    #nx = [(v,lookup[i]) for i,v in enumerate(sims)]

    #pprint(sorted(nx,reverse=True))


    ##tfidf = models.TfidfModel(corpus)
    ##tfidf = models.TfidfModel(corpus, id2word=dictionary, dictionary=dictionary)
    tfidf = models.TfidfModel(corpus, id2word=dictionary)

    #doc1 = a1.features[a].text
    #vec_bow1 = dictionary.doc2bow(doc1.lower().split())
    #vec_tfidf1 = tfidf[vec_bow1] # convert the query to tfidf space

    #doc2 = a1.features[b].text
    #vec_bow2 = dictionary.doc2bow(doc2.lower().split())
    #vec_tfidf2 = tfidf[vec_bow2] # convert the query to tfidf space

    index = similarities.MatrixSimilarity(tfidf[corpus])

    sims = index[vec_tfidf1]

    nx = [(v,lookup[i]) for i,v in enumerate(sims)]

    pprint(sorted(nx,reverse=True))




    #from analogy import kulczynski_2, jaccard_index
    
    #stoplist = set('for a of the and to in'.split())

    #doc1 = set(a1.features[a].text.lower().split()) - stoplist

    #sims = []

    #for f in a1.features.values():
    #    sims.append((kulczynski_2(set(f.text.lower().split())-stoplist,doc1),f.name))

    #pprint(sorted(sims,reverse=True))

    #from nltk.corpus import wordnet as wn

    #dog = wn.synset('dog.n.01')

    #cat = wn.synset('cat.n.01')

    #dog.path_similarity(cat)
    #dog.lch_similarity(cat)
    #dog.wup_similarity(cat)

    #pprint([(x,x.definition()) for x in wn.synsets('c++')])

def get_similarity(a1, a, corpus, dictionary, lookup):
    tfidf = models.TfidfModel(corpus, id2word=dictionary)
    doc = a1.features[a].text
    vec_bow1 = dictionary.doc2bow(doc.lower().split())
    index = similarities.MatrixSimilarity(tfidf[corpus])
    sims = index[tfidf[vec_bow1]]
    nx = [(v,lookup[i]) for i,v in enumerate(sims)]
    pprint(sorted(nx,reverse=True))



    