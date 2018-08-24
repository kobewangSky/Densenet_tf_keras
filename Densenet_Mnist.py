import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dropout, AvgPool2D, Concatenate, Dense, Activation, MaxPool2D, Flatten
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

growth_k = 12
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 10
batch_size = 100

total_epochs = 50




class DenseNet():
    def __init__(self, x, nb_blocks, filters, b_training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.b_training = b_training
        self.model = self.Dense_net(x)

    def Batch_Normalization(self, x, training, scope):
        return tf.cond(training,
                       lambda: batch_norm(scope=scope, inputs=x, is_training=training, reuse=None, updates_collections=None, decay=0.9, center=True, scale=True, zero_debias_moving_mean=True),
                       lambda: batch_norm(scope=scope, inputs=x, is_training=training, reuse=True, updates_collections=None, decay=0.9, center=True, scale=True, zero_debias_moving_mean=True))



    def bottleneck_layer(self, x, s_scope):
        with tf.name_scope(s_scope):

            x = self.Batch_Normalization(x, self.b_training, s_scope + '_batch_normal_0')
            x = Activation('relu', name=s_scope + '_relu1')(x)
            x = Conv2D(filters=4 * self.filters, kernel_size=1,padding='SAME', name=s_scope + '_conv_1')(x)
            x = Dropout(rate=dropout_rate, trainable=self.b_training)(x)


            x = self.Batch_Normalization(x, self.b_training, s_scope + '_batch_normal_1')
            x = Activation('relu', name=s_scope + '_relu2')(x)
            x = Conv2D(filters=4 * self.filters, kernel_size=3, padding='SAME', name=s_scope + '_conv_2')(x)
            x = Dropout(rate=dropout_rate, trainable=self.b_training)(x)
            return x

    def transition_layer(self, x, s_scope):
        with tf.name_scope(s_scope):
            x = self.Batch_Normalization(x, self.b_training, s_scope + '_batch_normal_0')
            x = Activation('relu', name=s_scope + '_relu1')(x)
            x = Conv2D(filters=4 * self.filters, kernel_size=1, padding='SAME', name=s_scope + '_conv_1')(x)
            x = Dropout(rate=dropout_rate, trainable=self.b_training)(x)
            x = AvgPool2D(pool_size=2, strides=2)(x)
            return x
    def dense_block(self, x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_list = list()
            layers_list.append(x)
            x = self.bottleneck_layer(x, s_scope= layer_name + '_bottleN_' + str(0))
            layers_list.append(x)

            for i in range(nb_layers - 1):
                x = Concatenate()(layers_list)
                x = self.bottleneck_layer(x, s_scope=layer_name + '_bottleN_' + str(i + 1))
                layers_list.append(x)

            x = Concatenate()(layers_list)
            return x

    def Dense_net(self, x):
        x = Conv2D(filters=2 * self.filters, kernel_size=7, strides=2, name='conv0')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        for i in range(self.nb_blocks):
            x = self.dense_block(x = x, nb_layers=4, layer_name='dense_' + str(i))
            x = self.transition_layer(x, s_scope='trans_'+str(i))

        x = self.dense_block(x=x, nb_layers=32, layer_name='Dense_finial')

        x = self.Batch_Normalization(x, self.b_training, 'batch_normal_finish')

        x = Activation('relu', name='relu_finish')(x)
        x = GlobalAvgPool2D()(x)
        x = Flatten()(x)
        x = Dense(units=class_num)(x)
        return x

x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])


label = tf.placeholder(tf.float32, shape=[None, 10])

b_isTrain = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, b_training=b_isTrain).model
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits))

optimizers = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = epsilon)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_ = optimizers.minimize(cost)



correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', acc)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    globals_step = 0
    epoch_learning_rate = init_learning_rate

    for epoch in range(total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(mnist.train.num_examples / batch_size)


        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            train_feed_dict = {
                x : batch_x,
                label : batch_y,
                b_isTrain : True,
                learning_rate : epoch_learning_rate
            }

            _, loss = sess.run([train_, cost], feed_dict=train_feed_dict)

            if step % 100 == 0:
                globals_step +=100
                train_summary, train_accuracy = sess.run([merged, acc], feed_dict=train_feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)

            test_feed_dict = {
                x: mnist.test.images,
                label: mnist.test.labels,
                learning_rate: epoch_learning_rate,
                b_isTrain: False
            }
        accuracy_rates = sess.run(acc, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
    saver.save(sess=sess, save_path='./model/dense.ckpt')