import numpy as np
import json
from gensim.models import KeyedVectors
import torch
def build_embedding_matrix(vocab_json="./data/clean_spacy/vocab.json", fasttext_path="./data/fasttext/cc.en.300.vec", out_matrix="./data/clean_spacy/embedding_matrix.npy"):
    with open(vocab_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["vocab"]
    word2idx = data["word2idx"]
    embedding_dim = 300  # adjust if using different model
    print("Loading FastText vectors ...")
    ft = KeyedVectors.load_word2vec_format(fasttext_path, binary=False)
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, ft.vector_size), dtype=np.float32)
    # random for <unk> and <sos>/<eos>
    rng = np.random.RandomState(1234)
    unk_vec = rng.normal(scale=0.6, size=(ft.vector_size,)).astype(np.float32)
    sos_vec = rng.normal(scale=0.2, size=(ft.vector_size,)).astype(np.float32)
    for word, idx in word2idx.items():
        if word == "<pad>":
            embedding_matrix[idx] = np.zeros((ft.vector_size,), dtype=np.float32)
        elif word == "<unk>":
            embedding_matrix[idx] = unk_vec
        elif word in ("<sos>", "<eos>"):
            embedding_matrix[idx] = sos_vec
        else:
            if word in ft:
                embedding_matrix[idx] = ft[word]
            else:
                # use FastText's subword based get_vector if available,
                # gensim KeyedVectors has get_vector for OOV subword composition sometimes.
                try:
                    embedding_matrix[idx] = ft.get_vector(word)
                except Exception:
                    embedding_matrix[idx] = unk_vec
    np.save(out_matrix, embedding_matrix)
    print(f"Saved embedding matrix to {out_matrix} (shape: {embedding_matrix.shape})")
    return out_matrix
