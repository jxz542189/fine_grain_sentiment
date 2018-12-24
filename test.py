from sklearn.externals import joblib
import os
import jieba
from collections import Counter
path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(path, 'data')
print(path)
# train_text = joblib.load(os.path.join(data_path, 'text_result_train.m'))
# train_text = [len(' '.join(jieba.cut(line)).split()) for line in train_text]
# validation_text = joblib.load(os.path.join(data_path, 'text_result_validation.m'))
# validation_text = [len(' '.join(jieba.cut(line)).split()) for line in validation_text]
# train_text.extend(validation_text)
# joblib.dump(train_text, os.path.join(data_path, 'text_length.m'))
# print(train_text)
text_length = joblib.load(os.path.join(data_path, 'text_length.m'))
print(Counter(text_length).keys())