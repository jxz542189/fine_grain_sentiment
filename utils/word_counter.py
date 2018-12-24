from sklearn.externals import joblib
import os
import jieba
from collections import Counter
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, 'data')
print(path)


def get_text_lengths():
    train_text = joblib.load(os.path.join(data_path, 'text_result_train.m'))
    train_text = [len(' '.join(jieba.cut(line)).split()) for line in train_text]
    validation_text = joblib.load(os.path.join(data_path, 'text_result_validation.m'))
    validation_text = [len(' '.join(jieba.cut(line)).split()) for line in validation_text]
    train_text.extend(validation_text)
    joblib.dump(train_text, os.path.join(data_path, 'text_length.m'))
    print(train_text)
    return train_text


def get_percent(percent=0.97):
    text_lengths = joblib.load(os.path.join(data_path, 'text_length.m'))
    total = 0
    c = Counter(text_lengths)
    for key in c.keys():
        total += c[key]
    text_lengths_sorted = sorted(c.items(), key=lambda item:item[1], reverse=True)
    current_length = 0
    for key, value in text_lengths_sorted:
        print(float(current_length + value) / total)
        if float(current_length + value) / total > percent:
            return key
        else:
            current_length += value
    # print(text_lengths_sorted)

print(get_percent())
# print(Counter(text_length).keys())