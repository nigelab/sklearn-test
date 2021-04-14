from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


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

def count_chinese_demo():

    data = ["我爱北京天安门", "天安门上太阳升"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
    print("data new toarray: \n", data_new.toarray())
    print("feature names: \n", transfer.get_feature_names())
    return None

def count_chinese_split_demo():

    data = ["我爱 北京 天安门", "天安门 上 太阳升"]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)

    print("data new: \n", data_new)
    print("data new toarray: \n", data_new.toarray())
    print("feature names: \n", transfer.get_feature_names())
    return None

if __name__ == "__main__":
    count_chinese_split_demo()
