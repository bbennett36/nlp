import nltk
from spacy.en import English
from nltk.tokenize import word_tokenize
parser = English()
english_stopwords = set(nltk.corpus.stopwords.words('english'))

print(english_stopwords)

text = 'The chicken went to the house to humiliate the man'
tokens = word_tokenize(text)
content_tokens = [token for token in tokens if token.lower() not in english_stopwords]
print(content_tokens)

# def print_token(token):
#     print('================')
#     print('value:', token.orth_)
#     print('lemma:', token.lemma_)
#     print('shape:', token.shape_)
#
# text = 'He ran to the store because he was kind of apes.'
# tokens = parser(text)
# for token in tokens:
#     print_token(token)