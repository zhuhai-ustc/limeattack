import numpy as np
from nltk.corpus import stopwords
def generate_embedding_mat(counter_fitting_embeddings_path):
        ''' Generate word2idx and idx2word dicts for quick lookup of words from indices of cosine similarity matrix and vice versa

        Args:
            counter_fitting_embeddings_path: Path to counter fitting embeddings
        '''
        print("Building vocab")
        idx2word_vocab = {}
        word2idx_vocab = {}
        i=0
        embeddings = []
        with open(counter_fitting_embeddings_path, 'r',encoding='utf-8') as ifile:
          for line in ifile:
            word = line.split()[0]
            if word not in idx2word_vocab:
              idx2word_vocab[len(idx2word_vocab)] = word
              word2idx_vocab[word] = len(idx2word_vocab) - 1

        with open(counter_fitting_embeddings_path, 'r',encoding='utf-8') as ifile:
          for line in ifile:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(np.array(embedding))
           
        embeddings=np.array(embeddings)
        print(type(embeddings))
        print(embeddings.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm)

        return embeddings, word2idx_vocab , idx2word_vocab



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

