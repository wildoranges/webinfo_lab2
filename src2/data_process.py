from gensim.models import word2vec
import numpy as np
import torch.nn as nn


def get_word_vec(entity_text_path="../dataset/entity_with_text.txt", relation_text_path="../dataset/relation_with_text.txt",
                 output_text_path="../dataset/des.txt", vector_size=100)->word2vec.Word2Vec:
    with open(entity_text_path, "r") as fr1, open(relation_text_path, "r") as fr2, open(output_text_path, "w+") as fw:
        for line in fr1.readlines():
            fw.write(line.strip().split("\t")[1])
            fw.write("\n")
            
        for line in fr2.readlines():
            fw.write(line.strip().split("\t")[1])
            fw.write("\n")
              
    sentences = word2vec.LineSentence(output_text_path)
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, vector_size=vector_size)
    return model

def get_h_r_vec(model:word2vec.Word2Vec, entity_text_path="../dataset/entity_with_text.txt", 
                relation_text_path="../dataset/relation_with_text.txt"):
    h = {}
    r = {}
    with open(entity_text_path, "r")  as f:
        for line in f.readlines():
            words = line.strip().split('\t')
            key = words[0]
            text = words[1].split(" ")
            vec = 0
            for word in text:
                vec += model.wv[word]
            vev = vec / np.linalg.norm(vec)
            h[key] = vec
            
    with open(relation_text_path, "r")  as f:
        for line in f.readlines():
            words = line.strip().split('\t')
            key = words[0]
            text = words[1].split(" ")
            vec = 0
            for word in text:
                vec += model.wv[word]
            vev = vec / np.linalg.norm(vec)
            r[key] = vec
            
    return h, r
    
    