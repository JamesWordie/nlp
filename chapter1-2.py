import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer

# Basic
sentence = """Thomas Jefferson began building Monticello at the age of 26."""
# print(sentence.split())

# Tokenizer
token_sequence = str.split(sentence) # quick and dirty
vocab = sorted(set(token_sequence)) # lists all UNIQUE tokens, sorted so number < letter, capital < lowercase
', '.join(vocab)

# print("token seq\n", token_sequence)
# print("vocab\n", vocab)


num_tokens = len(token_sequence)
vocab_size = len(vocab)
# print(num_tokens)
# print(vocab_size)

# empty table is as wide as count of unique vocab terms
# and as high as the length of the document, ie 10 x 10
onehot_vectors = np.zeros((num_tokens, vocab_size), int)

# for each word in sentence, mark column for that word in the vocab with 1
for i, word in enumerate(token_sequence):
  onehot_vectors[i, vocab.index(word)] = 1
  ' '.join(vocab)

# print("vector\n", onehot_vectors)

# Pandas Dataframe
# df = pd.DataFrame(onehot_vectors, columns=vocab)
# # Make 0 empty string DO NOT DO IN REALITY
# df[df == 0] = ''
# print("dataframe\n", df)

# Storing in a dictionary
sentence_bow = {}
for token in sentence.split():
  sentence_bow[token] = 1
sortedsent = sorted(sentence_bow.items())
# print(sortedsent)

# Pandas Series
dfSeries = pd.DataFrame(pd.Series(dict([(token, 1) for token in
  sentence.split()])), columns=['sent']).T
# print(dfSeries)

# More Sentences
sentences = """Thomas Jefferson began building Monticello at the age of 26.\n"""
sentences += """Construction was done mostly by local masons and carpenters.\n"""
sentences += "He moved into the South Pavilion in 1770.\n"
sentences += """Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."""
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
  corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())
df2 = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
# print(df2[df2.columns[:10]])

# Vector Dot Product
v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])
dotp = v1.dot(v2)
dotp2 = (v1 * v2).sum()
# print(dotp)
# print(dotp2)

# Matrix product
# print(np.matmul(v1, v2))

# Equivalent to the above
# matprod = v1.reshape(-1, 1).T @ v2.reshape(-1,1)
# print(matprod)

# Overlapping word count
df = df2.T
# print("0-1", df.sent0.dot(df.sent1))
# print("0-2", df.sent0.dot(df.sent2))
# print("0-3", df.sent0.dot(df.sent3))

'''VSM - Vector Space Model'''
# print([(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v])

# Improving Tokenize
sentence = """Thomas Jefferson began building Monticello at the age of 26."""
tokens = re.split(r'[-\s.,;!?]+', sentence) #splits sentence on whitespace or punctuation that occurs at least once
# print(tokens)

pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
impvd = [x for x in tokens if x and x not in '- \t\n.,;!?']
# print(impvd)
# ['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26']

'''Tokenizer packages
-> spaCy—Accurate , flexible, fast, Python
-> Stanford CoreNLP—More accurate, less flexible, fast, depends on Java 8
-> NLTK—Standard used by many NLP contests and comparisons, popular, Python
NLTK easiest to use setup-wise
'''

# NLTK Package
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
# print(tokenizer.tokenize(sentence))

# NLTK TreeBank
sentence = """Monticello wasn't designated as UNESCO World Heritage Site until 1987."""
tokenizer = TreebankWordTokenizer()
# print(tokenizer.tokenize(sentence))

sentence = """Thomas Jefferson began building Monticello at the age of 26."""
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']

ngrams2 = list(ngrams(tokens, 2))
ngrams3 = list(ngrams(tokens, 3))
# print(ngrams2)
# [('Thomas', 'Jefferson'), ('Jefferson', 'began'), ('began', 'building'), ('building', 'Monticello'), ('Monticello', 'at'), ('at', 'the'), ('the', 'age'), ('age', 'of'), ('of', '26')]
# print(ngrams3)
# [('Thomas', 'Jefferson', 'began'), ('Jefferson', 'began', 'building'), ('began', 'building', 'Monticello'), ('building', 'Monticello', 'at'), ('Monticello', 'at', 'the'), ('at', 'the', 'age'), ('the', 'age', 'of'), ('age', 'of', '26')]

# print([" ".join(x) for x in ngrams2])
# ['Thomas Jefferson', 'Jefferson began', 'began building', 'building Monticello', 'Monticello at', 'at the', 'the age', 'age of', 'of 26']

''' Normalizing
removing capitalizations, ie lowercasing everything, not very beneficial
double tokens however allows for more context
'''

''' Stemming - regex rules
-> If a word ends with more than one s, the stem is the word and the suffix is a blank string.
-> If a word ends with a single s, the stem is the word without the s and the suffix is the s.
-> If a word does not end on an s, the stem is the word and no suffix is returned.
Stemmers are only really used in large-scale information retrieval applications (keyword search).
'''

def stem(phrase):
  return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)[0][0].strip("'") for word in phrase.lower().split()])
# print(stem('houses'))
# house
# print(stem("Doctor House's calls"))
# doctor house call

stemmer = PorterStemmer()
tmp = ' '.join([stemmer.stem(w).strip("'") for w in "dish washer's washed dishes".split()])
# print(tmp)
# dish washer wash dish

'''Lemmatization
For example “chat,” “chatter,” “chatty,” “chat- ting,” and perhaps even “chatbot” would all be treated the same in an NLP pipeline with lemmatization
Lemmatization is a potentially more accurate way to normalize a word than stem- ming or case normalization because it takes into account a word’s meaning
Lemmatizers are better than stemmers for most applications
'''

'''
When should you use a lemmatizer or a stemmer? Stemmers are generally faster to
compute and require less-complex code and datasets. But stemmers will make more
errors and stem a far greater number of words, reducing the information content
or meaning of your text much more than a lemmatizer would. Both stemmers and
lem- matizers will reduce your vocabulary size and increase the ambiguity of
the text. But lemmatizers do a better job retaining as much of the information
content as possible based on how the word was used within the text and its
intended meaning.
'''
