from keras.datasets import mnist

from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.contrib.tensor_forest.client import random_forest

from tensorflow.contrib.learn.python.learn.estimators import estimator

from tensorflow.contrib.learn.python.learn import metric_spec

from tensorflow.contrib.tensor_forest.client import eval_metrics


def rf_train(x_train, y_train, x_test, y_test):
    params = tensor_forest.ForestHParams(

        num_classes=10,

        num_features=784,

        num_trees=100,

        max_nodes=10000

    )

    graph_builder_class = tensor_forest.TrainingLossForest

    est = estimator.SKCompat(

        random_forest.TensorForestEstimator(

            params,

            graph_builder_class=graph_builder_class,

            model_dir="./models"

        )

    )

    est.fit(x=x_train, y=y_train, batch_size=128)

    metric_name = "accuracy"

    metric = {

        metric_name: metric_spec.MetricSpec(

            eval_metrics.get_metric(metric_name),

            prediction_key=eval_metrics.get_prediction_key(metric_name)

        )

    }

    results = est.score(x=x_test, y=y_test, batch_size=128, metrics=metric)

    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32")

    x_test = x_test.astype("float32")

    x_train = x_train.reshape(len(x_train), 784)

    x_test = x_test.reshape(len(x_test), 784)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()

    rf_train(x_train, y_train, x_test, y_test)
