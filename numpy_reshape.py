import numpy
x = [1, 2, 3]
y = [1618290457000, 1618290758000, 1618291060000]
print(x, y)
x_reshape = numpy.array(x).reshape(-1,1)
y_reshape = numpy.array(y).reshape(-1,1)
# x_reshape1 = numpy.array(x).reshape(len(x),1)
print(x_reshape)
print(y_reshape)


from sklearn import linear_model
#首先定义一个线性回归对象
lr=linear_model.LinearRegression()
#定义数据集
x=np.array([150,200,250,300,350,400,600])
y=np.array([6450,7450,8450,9450,11450,15450,18450])
#训练模型
lr.fit(square.reshape(-1,1),price)
#未完待续...