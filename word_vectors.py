import gensim.models.word2vec as word2vec
google_vec_file = '/home/brennan/dumps/GoogleNews-vectors-negative300.bin'
model = word2vec.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)



list(model.vocab.keys())[:10]

model['mouse']

model.most_similar['mouse']