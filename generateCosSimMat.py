import numpy as np
from nltk.corpus import stopwords

def csim_matrix(lst_revs ,embeddings, word2idx_vocab):
  ''' Create a cosine similarity matrix only for words in reviews

  Identify all the words in the reviews that are not stop words
  Use embedding matrix to create a cosine similarity submatrix just for these words
  Columns of this cosine similarity matrix correspond to the original words in the embedding
  rows of the matrix correspond to the words in the reviews
  '''
  # nlp = spacy.load('en_core_web_sm') # en
  reviews = [d[0] for d in lst_revs]

  all_words = set()
  
  for r in reviews:
    all_words = all_words.union(set([str(word) for word in r if str(word) in word2idx_vocab and str(word) not in stopwords.words('english')]))
  
  word2idx_rev={}
  idx2word_rev={}
  p=0
  embeddings_rev_words=[]
  for word in all_words:
    word2idx_rev[str(word)] = p
    idx2word_rev[p]=str(word)
    p+=1
    embeddings_rev_words.append(embeddings[word2idx_vocab[str(word)]])

  embeddings_rev_words=np.array(embeddings_rev_words)
  cos_sim = np.dot(embeddings_rev_words, embeddings.T)

  return cos_sim, word2idx_rev,idx2word_rev
