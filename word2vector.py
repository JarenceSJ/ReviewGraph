import os
from collections import defaultdict
from load_data import load_corpus, get_dir_and_base_name
import json
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

GLOVE_PATH = '/home/d1/shuaijie/data/glove.6B.100d.txt'
GLOVE_TMP_PATH = '/home/d1/shuaijie/data/glove.6B.100d.word2vec.txt'

# GLOVE_PATH = 'f:/data/glove.6B.100d.txt'
# GLOVE_TMP_PATH = 'f:/data/glove.6B.100d.word2vec.txt'


class Args:
    # dataset_name = 'Office_Products_5'
    # dataset_path = '/shuaijie/data/Office_Products_5/Office_Products_5.json'

    # dataset_name = 'Instant_Video_5'
    # dataset_path = '/home/d1/shuaijie/data/Instant_Video_5/Instant_Video_5.json'

    # dataset_name = 'Digital_Music_5'
    # dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'

    # dataset_name = 'Sports_and_Outdoors_5'
    # dataset_path = '/home/d1/shuaijie/data/Sports_and_Outdoors_5/Sports_and_Outdoors_5.json'

    dataset_name = 'Clothing_5'
    # dataset_path = '/home/d1/shuaijie/data/Clothing_5/Clothing_5.json'
    dataset_path = '/home/d1/shuaijie/NeuralEDUSeg/data/Clothing_5/Clothing_5.json'

    # dataset_name = 'Toys_and_Games_5'
    # dataset_path = '/home/d1/shuaijie/data/Toys_and_Games_5/Toys_and_Games_5.json'

    # dataset_name = 'Health_and_Personal_Care_5'
    # dataset_path = '/home/d1/shuaijie/data/Health_and_Personal_Care_5/Health_and_Personal_Care_5.json'

    # dataset_name = 'CDs_and_Vinyl_5'
    # dataset_path = '/home/d1/shuaijie/data/CDs_and_Vinyl_5/CDs_and_Vinyl_5.json'

    # dataset_name = 'Movies_and_TV_5'
    # dataset_path = '/home/d1/shuaijie/data/Movies_and_TV_5/Movies_and_TV_5.json'

    # dataset_name = 'Baby_5'
    # dataset_path = '/home/d1/shuaijie/data/Baby_5/Baby_5.json'

    # dataset_name = 'Yelp2013'
    # dataset_path = '/home/d1/shuaijie/data/yelp-recsys-2013/yelp2013.json'

    vocab_size = 50000
    embedding_dim = 100

    args_str = 'embed_dim_{}'.format(embedding_dim)


dir_path, _ = get_dir_and_base_name(Args.dataset_path)

Args.word2id_path = \
    '{}/word2id_{}.json'.format(dir_path, Args.args_str)
Args.embedding_path = \
    '{}/word_embedding_{}.npy'.format(dir_path, Args.args_str)


def load_glove():
    tmp_file = get_tmpfile(GLOVE_TMP_PATH)

    if not os.path.exists(GLOVE_TMP_PATH):
        glove_file = datapath(GLOVE_PATH)

        _ = glove2word2vec(glove_file, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file)

    # word2id = {word: index for index, word in enumerate(model.index2word)}
    word2id = model.key_to_index
    embed = model.vectors

    return word2id, embed


if __name__ == '__main__':
    args = Args
    print('Load Word2Vec from Glove, on dataset {}'.format(args.dataset_name))

    sentences = load_corpus(args.dataset_path)
    sentences = [x.split() for x in sentences]

    corpus_word_num = 0
    corpus_word_counter = defaultdict(int)
    for words in sentences:
        corpus_word_num += len(words)
        for word in words:
            corpus_word_counter[word] += 1

    corpus_word_counter = sorted(corpus_word_counter.items(),
                                 key=lambda x: x[1], reverse=True)
    corpus_word_counter = corpus_word_counter[:args.vocab_size]
    corpus_word_counter = dict(corpus_word_counter)

    glove_word2id, glove_embed = load_glove()
    corpus_word_set = set(corpus_word_counter.keys())
    glove_word_set = set(glove_word2id.keys())

    used_words = list(corpus_word_set & glove_word_set)
    unused_words = list(corpus_word_set - glove_word_set)

    unused_words_counter = {k: corpus_word_counter[k] for k in unused_words}
    unused_words_counter = dict(sorted(unused_words_counter.items(),
                                       key=lambda x: x[1], reverse=True))

    ds_word2id = {'<PAD>': 0}
    ds_embed = list()
    ds_embed.append(np.zeros(glove_embed.shape[1]))

    # used words
    for w in used_words:
        ds_word2id[w] = len(ds_word2id)
        ds_embed.append(glove_embed[glove_word2id[w]])

    for w in unused_words:
        ds_word2id[w] = len(ds_word2id)
        ds_embed.append(np.random.normal(.0, 1., glove_embed.shape[1]))

    ds_embed = np.stack(ds_embed, axis=0)

    with open(args.word2id_path, 'w') as f:
        json.dump(ds_word2id, f)
        
    np.save(args.embedding_path, ds_embed)
