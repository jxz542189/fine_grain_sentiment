import pandas as pd
import os
from sklearn.externals import joblib
import jieba
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, 'data')
column_names = ["location_traffic_convenience", "location_distance_from_business_district", "location_easy_to_find",
                "service_wait_time", "service_waiters_attitude", "service_parking_convenience",
                "service_serving_speed", "price_level", "price_cost_effective", "price_discount", "environment_decoration",
                "environment_noise", "environment_space", "environment_cleaness", "dish_portion",
                "dish_taste", "dish_look", "dish_recommendation", "others_overall_experience", "others_willing_to_consume_again"]
label2id = {}
label2onehot = {-2:[1,0,0,0], -1:[0,1,0,0], 0: [0,0,1,0], 1:[0,0,0,1]}
PAD = '<pad>'
UNK = '<unk>'
VOCAB_SIZE = 65679


def get_text_and_label(csv_name, type='train'):
    dataframe = pd.read_csv(csv_name)
    data_size = len(dataframe['content'])
    text_result = []
    label_result = []
    for i in range(data_size):
        # text = ''
        label = []
        text = dataframe['content'][i][1:-1]
        for column in column_names:
            label.extend(label2onehot[int(dataframe[column][i])])
        text_result.append(text)
        label_result.append(label)
    joblib.dump(text_result, os.path.join(data_path, 'text_result_{}.m'.format(type)))
    joblib.dump(label_result, os.path.join(data_path, 'label_result_{}.m'.format(type)))

    return text_result, label_result


def get_text(output_filename):
    train_text = joblib.load(os.path.join(data_path, 'text_result_train.m'))
    train_text = [' '.join(jieba.cut(line)) + '\n' for line in train_text]
    validation_text = joblib.load(os.path.join(data_path, 'text_result_validation.m'))
    validation_text = [' '.join(jieba.cut(line)) + '\n' for line in validation_text]
    train_text.extend(validation_text)
    with open(output_filename, 'w') as f:
        f.writelines(train_text)


def load_vec(vec_filename):
    word2vec = {}
    with open(vec_filename) as f:
        f.readline()
        i = 0
        dim = 0
        for line in f:
            line = line.split()
            if i == 0:
                dim = len(line[1:])
            if len(line[1:]) != dim:
                continue
            word2vec[line[0]] = [float(a) for a in line[1:]]
            i += 1
    joblib.dump(word2vec, os.path.join(data_path, 'word2vec.m'))
    return word2vec

def get_word2id(word2vec):
    word2id = {}
    word2id[PAD] = 0
    word2id[UNK] = 1
    for i, word in enumerate(word2vec.keys()):
        word2id[word] = i + 2
    joblib.dump(word2id, os.path.join(data_path, 'word2id.m'))
    return word2id


def get_train_or_test_data(type='train', length=500):
    text_result = joblib.load(os.path.join(data_path, 'text_result_{}.m'.format(type)))
    label_result = joblib.load(os.path.join(data_path, 'label_result_{}.m'.format(type)))
    label_result = [np.array(a) for a in label_result]
    word2id = joblib.load(os.path.join(data_path, 'word2id.m'))
    assert len(text_result) == len(label_result)
    data_res = []
    data_length = []
    for line in text_result:
        data = []
        words = ' '.join(jieba.cut(line.strip())).split()
        for word in words:
            if word in word2id:
                data.append(word2id[word])
            else:
                data.append(word2id[UNK])
        tmp_length = length if len(data) >= length else len(data)
        data_length.append(tmp_length)
        data = data[:length] if len(data) >= length else data + [0] * (length - len(data))
        data_res.append(np.array(data))
    joblib.dump(np.array(data_res), os.path.join(data_path, '{}_data_{}.m'.format(type, length)))
    joblib.dump(np.array(label_result), os.path.join(data_path, '{}_label_{}.m'.format(type, length)))
    joblib.dump(np.array(data_length), os.path.join(data_path, '{}_length_{}.m'.format(type, length)))
    return np.array(data_res), np.array(label_result)


def get_default_embedding():
    word2id = joblib.load(os.path.join(data_path, 'word2id.m'))
    word2vec = joblib.load(os.path.join(data_path, 'word2vec.m'))
    id2word = {}
    for key, value in word2id.items():
        id2word[value] = key
    tokens_embedding = []
    token_dim = len(word2vec['的'])
    print("token_dim: ", token_dim)
    start_embedding = np.random.randn(2, token_dim).astype(np.float32) / np.sqrt(token_dim)
    tokens_embedding.extend(start_embedding.tolist())
    for i in range(2, len(word2vec) + 2):
        vec = word2vec[id2word[i]]
        tokens_embedding.append(vec)
    tokens_embedding = [np.array(emb) for emb in tokens_embedding]
    joblib.dump(np.array(tokens_embedding), os.path.join(data_path, 'tokens_embedding.m'))
    return np.array(tokens_embedding)


if __name__ == '__main__':

    #=================测试load_vec和get_word2id===================
    # word2vec = load_vec(os.path.join(data_path, 'vec.txt'))
    # word2id = get_word2id(word2vec)
    # print(word2id)

    #==================测试get_train_or_test_data================
    # data, label = get_train_or_test_data()
    # get_train_or_test_data(type='validation')
    # print(data.shape)

    # ==================测试get_default_embedding=================
    tokens_embedding = get_default_embedding()
    print(tokens_embedding.shape)

    # get_text_and_label(os.path.join(data_path, 'sentiment_analysis_trainingset.csv'))
    # get_text_and_label(os.path.join(data_path, 'sentiment_analysis_validationset.csv'), type='validation')
    # get_text(os.path.join(data_path, 'text.txt'))

