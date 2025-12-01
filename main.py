import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# import gzip
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载数据
# with gzip.open("mnist.pkl.gz", "rb") as f:
#     (training_data, _ , _) = pickle.load(f, encoding="latin1")
#
# X_train, y_train = training_data
#
# # 第 0 张图片
# img = X_train[0].reshape(28, 28)
# label = y_train[0]

plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
