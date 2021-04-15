import numpy
x = [1, 2, 3]
y = [1618290457000, 1618290758000, 1618291060000]
print(x, y)
x_reshape = numpy.array(x).reshape(-1,1)
print(x_reshape, y)
