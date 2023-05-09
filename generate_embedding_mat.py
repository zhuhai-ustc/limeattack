import numpy as np
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

