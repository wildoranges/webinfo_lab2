from gensim.models import word2vec
import numpy as np


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
    model = word2vec.Word2Vec(sentences, min_count=1, window=10, vector_size=vector_size)
    model.save("../output/word2vec.model")
    return model

def get_h_r_vec(model:word2vec.Word2Vec, entity_text_path="../dataset/entity_with_text.txt", 
                relation_text_path="../dataset/relation_with_text.txt", norm=False):
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
            if norm:
                vec = vec / np.linalg.norm(vec)
            h[key] = vec
            
    with open(relation_text_path, "r")  as f:
        for line in f.readlines():
            words = line.strip().split('\t')
            key = words[0]
            text = words[1].split(" ")
            vec = 0
            for word in text:
                vec += model.wv[word]
            if norm:
                vec = vec / np.linalg.norm(vec)
            r[key] = vec
            
    return h, r
    
def load_and_process_data(train_filename='../dataset/train.txt', test_filename='../dataset/test.txt'):

    train_xy = np.loadtxt(train_filename, dtype=str)
    train_feature = train_xy[:, 0:-1]
    train_label = train_xy[:, [-1]]

    test_xy = np.loadtxt(test_filename, dtype=str)
    test_feature = test_xy[:, 0:-1]
    test_label = test_xy[:, [-1]]

    print("train_feature's shape:"+str(train_feature.shape))
    print("test_feature's shape:"+str(test_feature.shape))
    print('train_label\'s shape:{}'.format(train_label.shape))
    print("test_label's shape:{}".format(test_label.shape))

    return train_feature,train_label,test_feature,test_label 

def remove_no_description_token(e_dict, r_dict, input_path:str, output_path:str):
    ft = open(input_path, 'r')
    fw = open(output_path, 'w+')
    for line in ft.readlines():
        h, r, t = line.strip().split('\t')
        if (h not in e_dict.keys()) or (r not in r_dict.keys()) or (t not in e_dict.keys()):
            continue
        fw.write(line)
    ft.close()
    fw.close()