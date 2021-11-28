import numpy as np

def load_and_process_data(train_filename, test_filename):

    train_xy = np.loadtxt(train_filename, dtype=int)
    train_feature = train_xy[:, 0:-1]
    train_label = train_xy[:, [-1]]

    test_xy = np.loadtxt(test_filename, dtype=int)
    test_feature = test_xy[:, 0:-1]
    test_label = test_xy[:, [-1]]

    print("train_feature's shape:"+str(train_feature.shape))
    print("test_feature's shape:"+str(test_feature.shape))
    print('train_label\'s shape:{}'.format(train_label.shape))
    print("test_label's shape:{}".format(test_label.shape))

    return train_feature,train_label,test_feature,test_label 


if __name__ == "__main__":
    train_feature, train_label, test_feature, test_label = load_and_process_data('../dataset/train.txt', '../dataset/dev.txt')

