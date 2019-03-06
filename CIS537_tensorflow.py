import numpy as np
import tensorflow as tf
import _pickle as pickle
from sklearn.metrics import roc_auc_score

tf.logging.set_verbosity(tf.logging.INFO)

def conv_classifier(features,labels,mode):
    # Input Layer
    # TODO check the size
    input_layer = tf.reshape(features["x"], [-1, 34, 26, 29])

    # Convolutional Layer #1
    # Applies 32 5x5 filters (extracting 5x5-pixel subregions), with tanh activation function
    # TODO check filter and kernel size
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)

    # Pooling Layer #1
    # Performs max pooling with a 2x2 filter and stride of 2
    # (which specifies that pooled regions do not overlap)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # Applies 64 5x5 filters, with tanh activation function
    # TODO check filter and kernel size
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.tanh)

    # Again, performs max pooling with a 2x2 filter and stride of 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 6 * 64])

    # 1,024 neurons, with dropout regularization rate of 0.4
    # (probability of 0.4 that any given element will be
    # dropped during training)
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.tanh)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # 2 neurons, one for positive and one for negative
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def evaluate_lenet5():

    f = open('datastage2_all.p', 'rb')
    data = pickle.load(f)
    f.close()

    data = list(data)

    all_data = [a[0] for a in data]
    all_labels = [a[1] for a in data]

    all_data = np.asarray(all_data)
    all_labels = np.asarray(all_labels)

    #460 controls, 115 cases

   # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
   # iter = dataset.make_initializable_iterator()
   # features, labels = iter.get_next()


    train_data = all_data[0:300]
    train_labels = all_labels[0:300]

    eval_data = all_data[300:]
    eval_labels = all_labels[300:]


    classifier = tf.estimator.Estimator(
        model_fn=conv_classifier, model_dir="model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    # TODO AUC
    # a = tf.Variable(eval_labels)
    # b = tf.Variable(eval_results)
    # auc = tf.contrib.metrics.streaming_auc(a, b)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    # sess.run(tf.initialize_local_variables()) # try commenting this line and you'll get the error
    # train_auc = sess.run(auc)
    # print(train_auc)
    
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     _,roc_score = tf.metrics.auc(eval_labels, eval_results)
    #     print(sess.run(roc_score, feed_dict={x : eval_data, y : eval_labels, p_keep_input: 1.0, p_keep_hidden: 1.0}))
    
    
    print(eval_results)


if __name__ == '__main__':
    evaluate_lenet5()