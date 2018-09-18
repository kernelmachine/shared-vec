import logging
import argparse
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm, trange

def linear_cca(H1, H2, outdim_size=300):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B


def _load_word_vectors_from_file(fname):
        res = dict(word_vecs={},
                   vocab_size={},
                   vector_size={},
                   index2word={})
        skipped = 0
        with open(fname, 'r') as f:
            header = f.readline()
            vocab_size, vector_size = tuple(map(int, header.split()))
            res['vocab_size'] = vocab_size
            res['vector_size'] = vector_size
            for i, line in tqdm(enumerate(f), total=vocab_size):
                line = line.rstrip().split()
                word, vector = line[0], line[1:]
                if len(vector) == vector_size:
                    try:
                        vector = list(map(float, vector))
                        res['word_vecs'][word] = vector
                    except ValueError:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue
            logger.info("Skipped {} words in file.".format(skipped))
            res['vocab_size'] = len(res['word_vecs'])

        for i, word in enumerate(res['word_vecs'].keys()):
                res['index2word'][i] = word
        res['embedding_matrix'] = np.zeros((res['vocab_size'],
                                            res['vector_size']))
        for i in range(res['vocab_size']):
            word = res['index2word'][i]
            embedding_vector = res['word_vecs'].get(word)
            if embedding_vector is not None:
                res['embedding_matrix'][i, :] = embedding_vector
        return res

def get_aligned_vectors(wordAlignFile, src_word_vecs, target_word_vecs):
    alignedVectors = {}
    for line in open(wordAlignFile, 'r'):
        lang1Word, lang2Word = line.strip().split(" ||| ")
        if lang2Word not in target_word_vecs['word_vecs'] or lang1Word not in src_word_vecs['word_vecs']: 
            continue
        else:
            alignedVectors[lang2Word] = src_word_vecs['word_vecs'][lang1Word]
    aligned_src = np.stack([alignedVectors[word] for word in alignedVectors])
    aligned_target = np.stack([target_word_vecs['word_vecs'][word] for word in alignedVectors])
    return aligned_src, aligned_target

if __name__ == "__main__":
    par = argparse.ArgumentParser(description='parser')
    par.add_argument('--alignment', '-a', type=str)
    par.add_argument('--src', '-s', type=str)
    par.add_argument('--tgt', '-t', type=str)
    par.add_argument('--out', '-o', type=str)
    args = par.parse_args()
    np.seterr(all='raise')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading word vectors from {}...".format(args.src))
    src_word_vecs = _load_word_vectors_from_file(args.src)
    logger.info("Loading word vectors from {}...".format(args.tgt))
    target_word_vecs = _load_word_vectors_from_file(args.tgt)
    logger.info("Retrieving alignment from {}...".format(args.alignment))
    aligned_src, aligned_target = get_aligned_vectors(args.alignment, src_word_vecs, target_word_vecs)
    src_word_vecs['embedding_matrix'] = normalize(src_word_vecs['embedding_matrix'])
    target_word_vecs['embedding_matrix'] = normalize(target_word_vecs['embedding_matrix'])
    aligned_src = normalize(aligned_src)
    aligned_target = normalize(aligned_target)
    logger.info("Projecting {} to {}...".format(args.src, args.tgt))
    cca = CCA(n_components=src_word_vecs['vector_size'])
    src_word_vecs['projected_matrix'], target_word_vecs['projected_matrix'] = linear_cca(aligned_src, aligned_target)
    src_to_target = np.dot(src_word_vecs['projected_matrix'], target_word_vecs['projected_matrix'].T)
    src_in_target = np.dot((src_word_vecs['embedding_matrix'] - np.tile(np.mean(src_word_vecs['embedding_matrix'], axis=0), 1)), src_to_target)
    src_in_target = normalize(src_in_target)
    logger.info("Writing to {}...".format(args.out))

    with open(args.out, 'w+') as f:
        f.write("{} {}\n".format(src_word_vecs['vocab_size'],
                               src_word_vecs['vector_size']))
        text = ""
        for ix in trange(src_word_vecs['vocab_size']):
            arr = " ".join([str(val) for val in src_in_target[ix, :].tolist()])
            text += "{} {}\n".format(src_word_vecs['index2word'][ix], arr)
            if ix % 10000 == 0:
                f.write(text)
                text = ""
    logger.info("Done!")
