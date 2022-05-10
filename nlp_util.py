import logging
import re
import numpy as np
from copy import copy

logger = logging.getLogger(__name__)


def clean_text_for_corpus(text, min_frequency_num=3, min_sent_length=3):
    """
    生成训练word2vector的语料
    """

    # 过滤无向量单词、停词，统计词频
    tokens_count = dict()
    corpus_list = []
    words_num = 0
    for sent in text:
        corpus = []

        for token in clean_str(sent).split():
            corpus.append(token)
            tokens_count[token] = tokens_count.setdefault(token, 0) + 1
            words_num += 1

        corpus_list.append(corpus)

    # low_frequency_num = int(words_num * 1e-4)
    # 删除低频词
    low_freq_tokens = {k for k, v in tokens_count.items()
                       if v < min_frequency_num}

    corpus_list = [[t for t in s if t not in low_freq_tokens]
                   for s in corpus_list]

    # 删除短句
    # corpus_list = [x for x in corpus_list if len(x) > min_sent_length]

    result = [' '.join(x) for x in corpus_list]

    return result


# def split_sentence_spacy(text: str):
#     if 'sentencizer' not in SpacyNLP().pipe_names:
#         SpacyNLP().add_pipe(SpacyNLP().create_pipe('sentencizer'))
#
#     doc = SpacyNLP()(text.lower())
#     result = [x.text for x in doc.sents]
#     return result


# def cut_sentence(text, max_length):
#     doc = SpacyNLP()(text)
#     if len(doc) > 0:
#         return doc[:max_length].text
#     else:
#         return ''

    
def sentence_to_token_id_list(sentence, word2id):
    tokens = sentence.split()
    token_list = []
    for t in tokens:
        try:
            token_list.append(word2id[t])
        except KeyError:
            pass
    return token_list


def filter_unused_words(sentence, word2id):
    tokens = sentence.split()
    token_list = []
    for t in tokens:
        if t in word2id:
            token_list.append(t)
    return ' '.join(token_list)


# def delete_uncommon_words_punct(sentence, del_stop_word=True):\
#
#     words = []
#     for token in SpacyNLP()(sentence):
#         if not token.is_alpha:
#             continue
#         if token.norm in SpacyNLP().vocab.vectors.key2row.keys():
#             if del_stop_word:
#                 if token.is_stop:
#                     continue
#
#             words.append(token.text)
#
#     result = ' '.join(words)
#     return result


def get_token_count(sentences):
    token_count = dict()
    for sent in sentences:
        for word in sent.split():
            token_count[word] = token_count.setdefault(word, 0) + 1

    return token_count


# def get_spacy_word_embedding() -> np.ndarray:
#     return SpacyNLP().vocab.vectors.data


def fuse_two_word_embedding(word2id_a, embed_a, word2id_b, embed_b):
    word2id = copy(word2id_a)
    embed = np.copy(embed_a)

    for k, v in word2id_b.items():
        if k not in word2id:
            word2id[k] = len(word2id)
            embed = np.vstack([embed, embed_b[v]])

    return word2id, embed


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9',.!;?()]", " ", string)

    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!+", " ! ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\\", " \\ ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)

    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"(\.|\s){7,}", " ... ", string)
    string = re.sub(r"(?<= )(\w \. )+(\w \.)", lambda x: x.group().replace(" ", ""), string)
    # string = re.sub(r"(\.|\s){4,}", " ... ", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)

    # string = re.sub(r"[^A-Za-z0-9']", " ", string)
    string = re.sub(r"(?!(('(?=s\b))|('(?=ve\b))|('(?=re\b))|('(?=d\b))|('(?=ll\b))|('(?=m\b))|((?<=n\b)'(?=t\b))))'", " ", string)

    # Glove style
    # string = re.sub(' [0-9]{5,} ', ' ##### ', string)
    # string = re.sub(' [0-9]{4} ', ' #### ', string)
    # string = re.sub(' [0-9]{3} ', ' ### ', string)
    # string = re.sub(' [0-9]{2} ', ' ## ', string)
    string = re.sub(' 0 ', ' zero ', string)
    string = re.sub(' 1 ', ' one ', string)
    string = re.sub(' 2 ', ' two ', string)
    string = re.sub(' 3 ', ' three ', string)
    string = re.sub(' 4 ', ' four ', string)
    string = re.sub(' 5 ', ' five ', string)
    string = re.sub(' 6 ', ' six ', string)
    string = re.sub(' 7 ', ' seven ', string)
    string = re.sub(' 8 ', ' eight ', string)
    string = re.sub(' 9 ', ' nine ', string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
