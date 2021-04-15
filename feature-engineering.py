from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def demo_load_iris():
    """
    sklearn datasets usage demo
    :return:
    """
    # load iris datasets
    iris = load_iris()
    # return a Bunch
    print("\nload iris:\n", iris)
    print("\nload iris desc:\n", iris.DESCR)
    print("\nload iris feature names:\n", iris.feature_names)
    print("\nload iris data:\n", iris['data'], iris.data.shape)

    # datasets split
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=22)
    print("train features: \n", x_train, x_train.shape)

    return None


def dict_demo():
    """
    dictionary feature collect
    :reutrn:
    """

    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    transferSparse = DictVectorizer()

    data_sparse = transferSparse.fit_transform(data)

    print("data sparse: \n", data_sparse)
#     data sparse:
#    (0, 1)       1.0
#   (0, 3)        100.0
#   (1, 0)        1.0
#   (1, 3)        60.0
#   (2, 2)        1.0
#   (2, 3)        30.0
    print("data sparse toarray: \n", data_sparse.toarray())
# [[  0.   1.   0. 100.]
#  [  1.   0.   0.  60.]
#  [  0.   0.   1.  30.]]

    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
#     data new:
#  [[  0.   1.   0. 100.]
#  [  1.   0.   0.  60.]
#  [  0.   0.   1.  30.]]
    print("feature names: \n", transfer.get_feature_names())
# feature names:
#  ['city=上海', 'city=北京', 'city=深圳', 'temperature']

    return None


def count_demo():

    data = ["life is short", "life is too long"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
    print("data new toarray: \n", data_new.toarray())
    print("feature names: \n", transfer.get_feature_names())
    return None
# data new:
#    (0, 1)       1
#   (0, 0)        1
#   (0, 3)        1
#   (1, 1)        1
#   (1, 0)        1
#   (1, 4)        1
#   (1, 2)        1
# data new toarray:
#  [[1 1 0 1 0]
#  [1 1 1 0 1]]
# feature names:
#  ['is', 'life', 'long', 'short', 'too']


def count_chinese_demo():

    data = ["我爱北京天安门", "天安门上太阳升"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
    print("data new toarray: \n", data_new.toarray())
    print("feature names: \n", transfer.get_feature_names())
    return None
# data new:
#    (0, 1)       1
#   (1, 0)        1
# data new toarray:
#  [[0 1]
#  [1 0]]
# feature names:
#  ['天安门上太阳升', '我爱北京天安门']


def count_chinese_split_demo():

    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
    print("data new toarray: \n", data_new.toarray())
    print("feature names: \n", transfer.get_feature_names())
    return None
# data new:
#    (0, 0)       1
#   (0, 1)        1
#   (1, 1)        1
#   (1, 2)        1
# data new toarray:
#  [[1 1 0]
#  [0 1 1]]
# feature names:
#  ['北京', '天安门', '太阳']


def cut_word(text):
    """
    split chinese words
    :param text:
    :return:
    """
    result = " ".join(list(jieba.cut(text)))
    # print(result)
    return result


def count_chinese_split_demo2():
    data = ["新闻——让每一次阅读都有价值！",
            "倾情打造专注新闻内容的新闻客户端，让更加优质、专业的新闻内容能够通过个性化方式分发每一位用户，满足用户获取关心的、感兴趣的新闻的需求。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    transfer = CountVectorizer(stop_words=["一个", "一位", "一条", "一次", "一篇"])

    data_final = transfer.fit_transform(data_new)

    print("data new: \n", data_final)
    print("data new toarray: \n", data_final.toarray())
    print("feature names: \n", transfer.get_feature_names())

    return None
# data new:
#    (0, 12)      1
#   (0, 20)       1
#   (0, 3)        1
#   (1, 12)       4
#   (1, 5)        1
#   (1, 11)       1
#   (1, 1)        1
#   (1, 7)        2
#   (1, 9)        1
#   (1, 14)       1
#   (1, 4)        1
#   (1, 0)        1
#   (1, 17)       1
#   (1, 19)       1
#   (1, 2)        1
#   (1, 13)       1
#   (1, 8)        1
#   (1, 16)       1
#   (1, 15)       1
#   (1, 18)       1
#   (1, 6)        1
#   (1, 10)       1
#   (1, 21)       1
# data new toarray:
#  [[0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0]
#  [1 1 1 0 1 1 1 2 1 1 1 1 4 1 1 1 1 1 1 1 0 1]]
# feature names:
#  ['专业', '专注', '个性化', '价值', '优质', '倾情', '关心', '内容', '分发', '客户端', '感兴趣', '打造', '新闻', '方式', '更加', '满足用户', '用户', '能够', '获取', '通过', '阅读', '需求']


if __name__ == "__main__":
    # cut_world("我爱北京天安门")
    count_demo()
    # count_chinese_demo()
    # count_chinese_split_demo()
    # count_chinese_split_demo2()
