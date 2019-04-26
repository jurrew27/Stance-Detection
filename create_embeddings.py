import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from feature_extractors import SentenceSplitter, TweetSplitter


train_data = pd.read_csv('data/train_clean.csv', escapechar='\\', encoding ='latin1')
test_data = pd.read_csv('data/test_clean.csv', escapechar='\\', encoding ='latin1')

tokenizer = SentenceSplitter()

train_tokens = tokenizer.transform(train_data['Tweet'])
all_tokens = train_tokens + tokenizer.transform(test_data['Tweet'])

train_docs = [TaggedDocument(tweet, [i]) for i, tweet in enumerate(tokenizer.transform(train_data['Tweet']))]
all_docs = train_docs + [TaggedDocument(tweet, [i]) for i, tweet in enumerate(tokenizer.transform(test_data['Tweet']))]

for size in (100, 200, 300):
    model = Word2Vec(train_tokens, size=size, min_count=1, workers=4, sg=1, iter=30)
    model.save('embeddings/word2vec-sg-{}.model'.format(size))

    model = Word2Vec(train_tokens, size=size, min_count=1, workers=4, sg=0, iter=30)
    model.save('embeddings/word2vec-cbow-{}.model'.format(size))

    model = Word2Vec(all_tokens, size=size, min_count=1, workers=4, sg=1, iter=30)
    model.save('embeddings/word2vec-sg-{}-all.model'.format(size))

    model = Word2Vec(all_tokens, size=size, min_count=1, workers=4, sg=0, iter=30)
    model.save('embeddings/word2vec-cbow-{}-all.model'.format(size))

    model = Doc2Vec(train_docs, vector_size=size, min_count=1, workers=4, dm=1, iter=30)
    model.save('embeddings/doc2vec-pv-dm-{}.model'.format(size))

    model = Doc2Vec(train_docs, vector_size=size, min_count=1, workers=4, dm=0, iter=30)
    model.save('embeddings/doc2vec-pv-dbow-{}.model'.format(size))

    model = Doc2Vec(all_docs, vector_size=size, min_count=1, workers=4, dm=1, iter=30)
    model.save('embeddings/doc2vec-pv-dm-{}-all.model'.format(size))

    model = Doc2Vec(all_docs, vector_size=size, min_count=1, workers=4, dm=0, iter=30)
    model.save('embeddings/doc2vec-pv-dbow-{}-all.model'.format(size))