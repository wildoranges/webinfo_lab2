import numpy as np

def load_and_process_data(train_filename, test_filename):

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

def load_text(entities_filename, relations_filename):
    entities_dict = {}
    relations_dict = {}
    sentences_e = []
    sentences_r = []

    with open(entities_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split('\t')
            entities_dict[line[0]] = line[1]
            sentences_e.append(line[1])

    with open(relations_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split('\t')
            relations_dict[line[0]] = line[1]
            sentences_r.append(line[1])

    word_list = ' '.join(sentences_e + sentences_r).split()
    word_list = list(set(word_list)) 
    word_dict = {w: i for i, w in enumerate(word_list)}
    index_dict = {i: w for i, w in enumerate(word_list)}


    return entities_dict, relations_dict, word_dict, index_dict

def get_max_seqlen(entities_dict: dict, relations_dict: dict):
    max_seqlen = 0
    for key, value in entities_dict.items():
        list = value.split()
        if len(list) > max_seqlen:
            max_seqlen = len(list)
            max_key = key
    
    for key, value in relations_dict.items():
        list = value.split()
        if len(list) > max_seqlen:
            max_seqlen = len(list)
            max_key = key

    return max_key, max_seqlen



if __name__ == "__main__":
    train_feature, train_label, test_feature, test_label = load_and_process_data('../dataset/train.txt', '../dataset/dev.txt')
    entities_dict, relations_dict = load_text('../dataset/entity_with_text.txt', '../dataset/relation_with_text.txt')
