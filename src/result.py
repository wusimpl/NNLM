import numpy as np
import math

"""Load file name via repositories

   Args
   ----
       voc_fname: str
            Vocabulary repository
       word_embeddings_fname: str
            np array, word embeddings repository

   Returns
   -------
       voc_file: numpy.array
            An numpy array contain chinese character or words
       word_embeddings_file: numpy.array
            An numpy array contain vectors
   """


def load(voc_fname, word_embeddings_fname):
    voc_file = np.load(voc_fname)
    word_embeddings_file = np.load(word_embeddings_fname)
    if len(voc_file) != len(word_embeddings_file):
        print("length are not the same!!!!voc:%d,embeddings:%d" % (len(voc_file), word_embeddings_file.shape[0]))
    return voc_file, word_embeddings_file


"""Implement cosine similarity in the range between [-1, 1]

    Args
    ----
      vector1: Vector 1
      vector2: Vector 2

    Returns
    -------
      numerator / denominator : cosine similarity of two vectors


    """


def cos_sim_numpy(vector1, vector2):
    numerator = sum(vector1 * vector2)
    denominator = math.sqrt(sum(vector1 ** 2) * sum(vector2 ** 2))
    return numerator / denominator


def run_similar_lst(search_word):
    '''Generate a sorted word similarity list

    Args
    ----
      search_word: a wanted search word
    Returns
    -------
      lst: ['word1': '1.0' , 'word2': '0.2143902']
    '''
    word_ind = voc_file.index(search_word)
    target_vector = f[word_ind]

    lst = list()
    for v in range(len(f)):
        lst.append((voc_file[v], cos_sim_numpy(f[v], target_vector)))
    return sorted(lst, key=lambda x: x[1], reverse=True)


def show(word, topn=10):
    """Find the top-N most similar words
    """
    print('\nshowing top {} words related {}: '.format(topn, word))
    for i in run_similar_lst(word)[:topn]:
        print('    {}, {}'.format(i[0], i[1]))


def save(fname_prefix, str_obj):
    with open('{}_similar_test.txt'.format(fname_prefix), 'w', encoding='utf-8') as wf:
        wf.write(str_obj)


if __name__ == '__main__':
    # Step 1 : load two files
    voc_file, f = load("data/vocab", "data/nnlm_word_embeddings.npy")
    vocab_length = len(voc_file)
    # In my opinion
    # Step 2 : show top-N similarity
    show("I", topn=100)
    show("am", topn=100)
    show("a", topn=100)

    # Step 3 : save it if you need
