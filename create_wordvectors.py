import gensim.downloader as api
from gensim.models import KeyedVectors

embedding = KeyedVectors.load_word2vec_format('../../Downloads/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
embedding.wv.save('embeddings_preloaded/word2vec-twitter-400.wv')
