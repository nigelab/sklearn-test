from sklearn.datasets import load_iris


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

    return None


if __name__ == "__main__":
    demo_load_iris()
