import nltk
import copy
import math
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from collections import OrderedDict
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer

'''
# DENSE PASSAGE RETRIEVAL - RETRIEVAL METHOD
'''


# import gensim.downloader as api

# # # print(list(gensim.downloader.info()['models'].keys()))
# # # ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']

# glove_vectors_wiki = api.load('glove-wiki-gigaword-200')

# # glove_vectors_twit = api.load('glove-twitter-25')

# # # mostsim = glove_vectors_twit.most_similar('twitter')

# # mostsim = glove_vectors_twit.most_similar('twitter')
# mostsim = glove_vectors_wiki.most_similar('twitter')

# print(mostsim)
# # # [('facebook', 0.948005199432373), ('tweet', 0.9403423070907593), ('fb', 0.9342358708381653), ('instagram', 0.9104824066162109), ('chat', 0.8964964747428894), ('hashtag', 0.8885937333106995), ('tweets', 0.8878158330917358), ('tl', 0.8778461217880249), ('link', 0.8778210878372192), ('internet', 0.8753897547721863)]

'''Importance
Bags of words—Vectors of word counts or frequencies
Bags of n-grams—Counts of word pairs (bigrams), triplets (trigrams), and so on
TF-IDF vectors—Word scores that better represent their importance
TF-IDF --> Term Frequency Inverse Document Frequency
'''


# BOW
sentence = """The faster Harry got to the store, the faster Harry, the faster, would get home."""
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
# print(tokens)

bag_of_words = Counter(tokens)
# print(bag_of_words)
# print(bag_of_words.most_common(4))

# Stop words and punctuation aren't helpful
times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears / num_unique_words
# print(round(tf, 4))

'''Reason for using TF like this, eg 2 docs one 30 words and one 500_000, 3 vs 100 occurrences
It helps to provide "context" roughly speaking '''

text = "A kite is traditionally a tethered heavier-than-air craft with wing surfaces that react against the air to create lift and drag. A kite consists of wings, tethers, and anchors. Kites often have a bridle to guide the face of the kite at the correct angle so the wind can lift it. A kite’s wing also may be so designed so a bridle is not needed; when kiting a sailplane for launch, the tether meets the wing at a single point. A kite may have fixed or moving anchors. Untraditionally in technical kiting, a kite consists of tether-set-coupled wing sets; even in technical kiting, though, a wing in the system is still often called the kite. The lift that sustains the kite in flight is generated when air flows around the kite’s surface, producing low pressure above and high pressure below the wings. The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached. The anchor point of the kite line may be static or moving (such as the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites or vehicle). The same principles of fluid flow apply in liquids and kites are also used under water. A hybrid tethered craft comprising both a lighter-than-air balloon as well as a kite lifting surface is called a kytoon. Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite fishing, kite buggying and a new trend snow kiting. Even Man-lifting kites have been made."
token_text = tokenizer.tokenize(text.lower())
token_counts = Counter(token_text)
# print(token_counts)

nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
tokens_stop = [x for x in token_text if x not in stopwords]
kite_counts = Counter(tokens_stop)
# print(kite_counts)

# Vectoring
document_vector = []
doc_length = len(tokens_stop)
# print('doc len', doc_length)
for key, value in kite_counts.most_common():
  document_vector.append(value / doc_length)

# print(document_vector)


docs = ["The faster Harry got to the store, the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")
doc_tokens = []
for doc in docs:
  doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
len(doc_tokens[0])
# 17
all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)
# 33
lexicon = sorted(set(all_doc_tokens))
len(lexicon)
# 18 - unique words across all 3 "documents (strings)"

zero_vector = OrderedDict((token, 0) for token in lexicon)
# print(zero_vector)

doc_vectors = []
for doc in docs:
  vec = copy.copy(zero_vector) # create a copy rather than a pointer, otherwise overwrites original
  tokens = tokenizer.tokenize(doc.lower())
  token_counts = Counter(tokens)
  for key, value in token_counts.items():
    vec[key] = value / len(lexicon)
  doc_vectors.append(vec)

# print(doc_vectors)

'''
Two vectors are “similar” if they share similar direction. They might have similar magnitude (length), which would mean that the word count (term frequency) vectors are for documents of about the same length
Cosine similarity is efficient to calculate because the dot product doesn’t require eval- uation of any trigonometric functions. In addition, cosine similarity has a convenient range for most machine learning problems: -1 to +1.
A·B=|A| |B|*cosΘ
Python -->  a.dot(b) == np.linalg.norm(a) * np.linalg.norm(b) / np.cos(theta)
For NLP document vectors that have a cosine similarity close to 1, you know that the documents are using similar words in similar proportion.

'''

# NLTK Brown Corpus
# nltk.download('brown')
# print(brown.words()[:10])

# print(brown.tagged_words()[:5])


puncs = set((',', '.', '--', '-', '!', '?',
  ':', ';', '``', "''", '(', ')', '[', ']'))
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
# print(token_counts.most_common(20))

'''
In short, if you rank the words of a corpus by the number of occurrences and
list them in descending order, you’ll find that, for a sufficiently large sample,
the first word in that ranked list is twice as likely to occur in the corpus as
the second word in the list. And it is four times as likely to appear as the
fourth word in the list. So given a large corpus, you can use this breakdown to
say statistically how likely a given word is to appear in any given document
of that corpus.
'''

# Kite - Wiki
kite_text = "A kite is traditionally a tethered heavier-than-air craft with wing surfaces that react against the air to create lift and drag. A kite consists of wings, tethers, and anchors. Kites often have a bridle to guide the face of the kite at the correct angle so the wind can lift it. A kite’s wing also may be so designed so a bridle is not needed; when kiting a sailplane for launch, the tether meets the wing at a single point. A kite may have fixed or moving anchors. Untraditionally in technical kiting, a kite consists of tether-set-coupled wing sets; even in technical kiting, though, a wing in the system is still often called the kite. The lift that sustains the kite in flight is generated when air flows around the kite’s surface, producing low pressure above and high pressure below the wings. The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached. The anchor point of the kite line may be static or moving (such as the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites or vehicle). The same principles of fluid flow apply in liquids and kites are also used under water. A hybrid tethered craft comprising both a lighter-than-air balloon as well as a kite lifting surface is called a kytoon. Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite fishing, kite buggying and a new trend snow kiting. Even Man-lifting kites have been made."
kite_history = "Kites were invented in China, where materials ideal for kite building were readily available: silk fabric for sail material; fine, high-tensile-strength silk for flying line; and resilient bamboo for a strong, lightweight framework. The kite has been claimed as the invention of the 5th-century BC Chinese philosophers Mozi (also Mo Di) and Lu Ban (also Gongshu Ban). By 549 AD paper kites were certainly being flown, as it was recorded that in that year a paper kite was used as a message for a rescue mission. Ancient and medieval Chinese sources describe kites being used for measuring distances, testing the wind, lifting men, signaling, and communication for military operations. The earliest known Chinese kites were flat (not bowed) and often rectangular. Later, tailless kites incorporated a stabilizing bowline. Kites were decorated with mythological motifs and legendary figures; some were fitted with strings and whistles to make musical sounds while flying. From China, kites were introduced to Cambodia, Thailand, India, Japan, Korea and the western world. After its introduction into India, the kite further evolved into the fighter kite, known as the patang in India, where thousands are flown every year on festivals such as Makar Sankranti. Kites were known throughout Polynesia, as far as New Zealand, with the assumption being that the knowledge diffused from China along with the people. Anthropomorphic kites made from cloth and wood were used in religious ceremonies to send prayers to the gods. Polynesian kite traditions are used by anthropologists get an idea of early “primitive” Asian traditions that are believed to have at one time existed in Asia."


kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)
intro_total = len(intro_tokens)
history_total = len(history_tokens)
# print(intro_total)
# 361 - word count
# print(history_total)
# 295 - word count

# Term Frequency
intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total

# print('Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite']))
# print('Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite']))

intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
# print('Term Frequency of "and" in intro is: {:.4f}'.format(intro_tf['and']))
# print('Term Frequency of "and" in history is: {:.4f}'.format(history_tf['and']))

'''A good way to think of a term’s inverse document frequency is this: How strange is it that this token is in this document? If a term appears in one document a lot of times, but occurs rarely in the rest of the corpus, one could assume it’s important to that document specifically. Your first step toward topic analysis!'''

'''A term’s IDF is merely the ratio of the total number of documents to the number of documents the term appears in.'''

# Rarity
num_docs_containing_and = 0
num_docs_containing_kite = 0
num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
  if 'and' in doc:
    num_docs_containing_and += 1
  if 'kite' in doc:
    num_docs_containing_kite += 1
  if 'china' in doc:
    num_docs_containing_china += 1

intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total
# print('Term Frequency of "china" in intro is: {:.4f}'.format(intro_tf['china']))
# print('Term Frequency of "china" in history is: {:.4f}'.format(history_tf['china']))

# IDF
num_docs = 2
intro_idf = {}
history_idf = {}
intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite
intro_idf['china'] = num_docs / num_docs_containing_china
history_idf['china'] = num_docs / num_docs_containing_china

# print('Inverse Document Frequency of "and" in intro is: {:.4f}'.format(intro_idf['and']))
# print('Inverse Document Frequency of "and" in history is: {:.4f}'.format(history_idf['and']))
# print('Inverse Document Frequency of "kite" in intro is: {:.4f}'.format(intro_idf['kite']))
# print('Inverse Document Frequency of "kite" in history is: {:.4f}'.format(history_idf['kite']))
# print('Inverse Document Frequency of "china" in intro is: {:.4f}'.format(intro_idf['china']))
# print('Inverse Document Frequency of "china" in history is: {:.4f}'.format(history_idf['china']))

# TFIDF
intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']
# print('TFIDF of "and" in intro is: {:.4f}'.format(intro_tfidf['and']))
# print('TFIDF of "kite" in intro is: {:.4f}'.format(intro_tfidf['kite']))
# print('TFIDF of "china" in intro is: {:.4f}'.format(intro_tfidf['china']))


history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']
# print('TFIDF of "and" in history is: {:.4f}'.format(history_tfidf['and']))
# print('TFIDF of "kite" in history is: {:.4f}'.format(history_tfidf['kite']))
# print('TFIDF of "china" in history is: {:.4f}'.format(history_tfidf['china']))

'''
And then finally, for a given term, t, in a given document, d, in a corpus, D, you get:

tf(t, d) = count(t) / count(d)

idf(t, D) = log ( number of documents / number of documents containing t )

tfidf(t, d, D) = tf(t, d) * idf(t, D)

So the more times a word appears in the document,
the TF (and hence the TF-IDF) will go up. At the same time,
as the number of documents that contain that word goes up,
the IDF (and hence the TF-IDF) for that word will go down.

TF-IDF, is the humble foundation of a simple search engine.
'''

# Back up to line ~100
document_tfidf_vectors = []
for doc in docs:
  vec = copy.copy(zero_vector)
  tokens = tokenizer.tokenize(doc.lower())
  token_counts = Counter(tokens)
  for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in docs:
      if key in _doc:
        docs_containing_key += 1
    tf = value / len(lexicon)
    if docs_containing_key:
      idf = len(docs) / docs_containing_key
    else:
      idf = 0
    vec[key] = tf * idf
  document_tfidf_vectors.append(vec)

# This gives a K-dimensional vector representation of each doc
# for x in document_tfidf_vectors:
  # print(x)
'''
Two vectors are considered similar if their cosine similarity is high
cosΘ = A·B / |A||B|
'''

# Basic TFIDF Search - cosine similarity
query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)
tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)
# print(token_counts)

for key, value in token_counts.items():
  docs_containing_key = 0
  for _doc in docs:
    if key in _doc.lower():
      docs_containing_key += 1
  if docs_containing_key == 0:
    continue
  tf = value / len(tokens)
  idf = len(docs) / docs_containing_key
  query_vec[key] = tf * idf

def cosine_sim(vec1, vec2):
  vec1 = [val for val in vec1.values()]
  vec2 = [val for val in vec2.values()]

  dot_prod = 0
  for i, v in enumerate(vec1):
    dot_prod += v * vec2[i]

  mag_1 = math.sqrt(sum([x**2 for x in vec1]))
  mag_2 = math.sqrt(sum([x**2 for x in vec2]))

  return dot_prod / (mag_1 * mag_2)

# print('vec1', query_vec)
# OrderedDict([
#   (',', 0),
#   ('.', 0),
#   ('and', 0),
#   ('as', 0),
#   ('faster', 0),
#   ('get', 0.2727272727272727),
#   ('got', 0),
#   ('hairy', 0),
#   ('harry', 0),
#   ('home', 0),
#   ('is', 0),
#   ('jill', 0),
#   ('not', 0),
#   ('store', 0.2727272727272727),
#   ('than', 0),
#   ('the', 0.2727272727272727),
#   ('to', 0.5454545454545454),
#   ('would', 0)
# ])
# print('vec2', document_tfidf_vectors[0])
# OrderedDict([
#   (',', 0.16666666666666666),
#   ('.', 0.05555555555555555),
#   ('and', 0.08333333333333333),
#   ('as', 0),
#   ('faster', 0.25),
#   ('get', 0.16666666666666666),
#   ('got', 0.16666666666666666),
#   ('hairy', 0),
#   ('harry', 0.0),
#   ('home', 0.16666666666666666),
#   ('is', 0),
#   ('jill', 0),
#   ('not', 0),
#   ('store', 0.16666666666666666),
#   ('than', 0),
#   ('the', 0.5),
#   ('to', 0.16666666666666666),
#   ('would', 0.16666666666666666)
# ])

# print(cosine_sim(query_vec, document_tfidf_vectors[0]))
# print(cosine_sim(query_vec, document_tfidf_vectors[1]))
# print(cosine_sim(query_vec, document_tfidf_vectors[2]))
# From results you can assume document 1 has the most relevance to the query
# 0.6132857433407973
# 0.0
# 0.0

# Do the above but using scipy and sklearn
corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
# print(model.todense().round(2))
'''TFIDF matrix of the tree docs
A matrix of your three documents and the inverse document frequency for each term in the lexicon
The TF-IDF of each term, token, or word in your lexicon make up the columns of the matrix (or again, the indices of each row)
'''
# [
#   [0.16 0. 0.48 0.21 0.21 0. 0.25 0.21 0. 0. 0. 0.21 0. 0.64 0.21 0.21]
#   [0.37 0. 0.37 0. 0. 0.37 0.29 0. 0.37 0.37 0. 0. 0.49 0. 0. 0.]
#   [0. 0.75 0. 0. 0. 0.29 0.22 0. 0.29 0.29 0.38 0. 0. 0. 0. 0.]
# ]
