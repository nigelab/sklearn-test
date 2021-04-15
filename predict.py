from sklearn.feature_extraction import DictVectorizer
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression():
    # X = [1, 2, 3]
    # Y = [1618290459000, 1618290458000, 1618290457000]
    data = pd.read_csv("./predict.csv")
    x = data['No']
    y = data['RequestTime']

    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    # 获取预测值
    predict_y = model.predict(x)

    # 构造返回字典
    predictions = {}
    predictions['intercept'] = model.intercept_  # 截距值
    predictions['coefficient'] = model.coef_    # 回归系数（斜率值）
    predictions['predict_value'] = predict_y

    # 绘出图像
    # 绘出已知数据散点图
    plt.scatter(x, y, color='blue')
    # 绘出预测直线
    plt.plot(x, predict_y, color='red', linewidth=2)

    plt.title('predict the request time')
    plt.xlabel('times')
    plt.ylabel('request time')
    plt.show()

    return None


def predict_next_request_time():
    """
    dictionary feature collect
    :reutrn:
    """

    data = [
        {'url': 'http://abc.com/site=1', 'request_time': 1618290459000,
            'response_time': 5000, 'x_tenant_id': 'amazon'},
        {'url': 'http://abc.com/site=2', 'request_time': 1618290458000,
            'response_time': 6000, 'x_tenant_id': 'amazon'},
        {'url': 'http://abc.com/site=3', 'request_time': 1618290457000,
            'response_time': 4000, 'x_tenant_id': 'amazon'},
        {'url': 'http://abc.com/site=11', 'request_time': 1618290458100,
            'response_time': 11000, 'x_tenant_id': 'wws'},
        {'url': 'http://abc.com/site=12', 'request_time': 1618290459200,
            'response_time': 7000, 'x_tenant_id': 'wws'},
    ]
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


if __name__ == "__main__":
    # predict_next_request_time()
    linear_regression()
