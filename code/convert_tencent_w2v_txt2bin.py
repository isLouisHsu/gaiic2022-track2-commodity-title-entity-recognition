from gensim.models import KeyedVectors

w2v_txt_model_zh_cn = "../data/public_data/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
vectors = KeyedVectors.load_word2vec_format(w2v_txt_model_zh_cn, binary=False)
w2v_bin_model_zh_cn = "../data/public_data/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"
vectors.save_word2vec_format(w2v_bin_model_zh_cn, binary=True)
