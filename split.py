from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    demo_load_iris()
