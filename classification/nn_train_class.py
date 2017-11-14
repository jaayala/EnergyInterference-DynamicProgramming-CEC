import tensorflow as tf
import numpy as np
import os.path
import random
import time
# import math

thres = 680

samplesDir = '../netsamples/'
samples_con = np.load(samplesDir+'consumption.npy')
samples_qos = np.load(samplesDir+'percentQoS.npy')
points = np.load(samplesDir+'points.npy')


samples_con2 = np.mean(samples_con, axis=1)

multfactor = 1e3


samples_qos2 = np.mean(samples_qos, axis=1) * multfactor
accepted_vals = samples_qos2 > thres
samples_qos2[accepted_vals] = 1.
samples_qos2[np.invert(accepted_vals)] = 0.

# randomizing the samples
print(str(len(points[:, 0]))+' total samples loaded')

random_index = np.arange(len(points[:, 0]))
random.shuffle(random_index)
points = points[random_index, :]
samples_con2 = samples_con2[random_index]
samples_qos2 = samples_qos2[random_index]

training_proportion = .8
tr_index = int(np.round(len(points[:, 0]) * training_proportion))

training_points = points[0:tr_index, :]
test_points = points[tr_index::, :]
training_con = samples_con2[0:tr_index]
test_con = samples_con2[tr_index::]
training_qos = samples_qos2[0:tr_index]
test_qos = samples_qos2[tr_index::]


# Parameters
con_learning_rate = 0.001
qos_learning_rate = 0.001
training_epochs = 400
batch_size = 50
display_step = 1

# Network Parameters
con_n_hidden_1 = 20  # 1st layer number of features
con_n_hidden_2 = 20  # 2nd layer number of features
qos_n_hidden_1 = 50  # 1st layer number of features
qos_n_hidden_2 = 50  # 2nd layer number of features
n_input = 4  # data input
n_classes = 1  # function output

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


con_weights = {
    'h1': tf.Variable(tf.random_normal([n_input, con_n_hidden_1]), name='h1c'),
    'h2': tf.Variable(tf.random_normal([con_n_hidden_1, con_n_hidden_2]), name='h2c'),
    'out': tf.Variable(tf.random_normal([con_n_hidden_2, n_classes]), name='houtc')
}

qos_weights = {
    'h1': tf.Variable(tf.random_normal([n_input, qos_n_hidden_1]), name='h1q'),
    'h2': tf.Variable(tf.random_normal([qos_n_hidden_1, qos_n_hidden_2]), name='h2q'),
    'out': tf.Variable(tf.random_normal([qos_n_hidden_2, n_classes]), name='houtq')
}
con_biases = {
    'b1': tf.Variable(tf.random_normal([con_n_hidden_1]), name='b1c'),
    'b2': tf.Variable(tf.random_normal([con_n_hidden_2]), name='b2c'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='boutc')
}
qos_biases = {
    'b1': tf.Variable(tf.random_normal([qos_n_hidden_1]), name='b1q'),
    'b2': tf.Variable(tf.random_normal([qos_n_hidden_2]), name='b2q'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='boutq')
}


# Construct model
pred_con = multilayer_perceptron(x, con_weights, con_biases)
cost_con = tf.reduce_sum(tf.pow(pred_con - y, 2)) / batch_size
cost_test_con = tf.reduce_sum(tf.pow(pred_con - y, 2)) / len(test_con)
optimizer_con = tf.train.AdamOptimizer(learning_rate=con_learning_rate).minimize(cost_con)


pred_qos = multilayer_perceptron(x, qos_weights, qos_biases)
predict_val = tf.greater(pred_qos, 0)
correct_prediction = tf.equal(tf.to_float(predict_val), y)
cost_qos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_qos, labels=y))
optimizer_qos = tf.train.AdamOptimizer(learning_rate=qos_learning_rate).minimize(cost_qos)

# Initializing the variables
init = tf.global_variables_initializer()


saver = tf.train.Saver()

# vCost_tr = np.zeros(training_epochs)
# vCost_test = np.zeros(training_epochs)
# vPred_tr = np.zeros(training_epochs)
# vPred_test = np.zeros(training_epochs)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    currentP = 0
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost_con = 0.
        avg_cost_qos = 0.
        total_batch = int(len(training_points[:, 0])/batch_size)
        currentP = 0
        nCorrectPrediction = 0

        # Loop over all batches
        for i in range(total_batch):

            batch_x = training_points[currentP:(currentP+batch_size), :]

            batch_y_con = training_con[currentP:(currentP+batch_size)]
            batch_y_con = np.expand_dims(batch_y_con, axis=1)

            batch_y_qos = training_qos[currentP:(currentP+batch_size)]
            batch_y_qos = np.expand_dims(batch_y_qos, axis=1)


            # Run optimization op (backprop) and cost op (to get loss value)
            _, c_con = sess.run([optimizer_con, cost_con], feed_dict={x: batch_x, y: batch_y_con})

            _, c_qos, cp = sess.run([optimizer_qos, cost_qos, correct_prediction], feed_dict={x: batch_x, y: batch_y_qos})

            # Compute average loss
            avg_cost_con += c_con / total_batch
            avg_cost_qos += c_qos / total_batch
            nCorrectPrediction += np.sum(cp)
            currentP += batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:

            c_test_con = sess.run([cost_test_con], feed_dict={x: test_points, y: np.expand_dims(test_con, axis=1)})
            c_test_con = c_test_con[0]
            c_test_qos, cp_test = sess.run([cost_qos, correct_prediction], feed_dict={x: test_points, y: np.expand_dims(test_qos, axis=1)})

            print("Epoch:", '%04d' % (epoch+1), "consumption_training_cost=", "{:.4f}".format(avg_cost_con))
            print("Epoch:", '%04d' % (epoch+1), "consumption_test_cost=", "{:.4f}".format(c_test_con))

            print("Epoch:", '%04d' % (epoch+1), "percentQoS_training_cost=", "{:.4f}".format(avg_cost_qos), '%_predict_training=', "{:.4f}".format(nCorrectPrediction/(total_batch*batch_size)))
            print("Epoch:", '%04d' % (epoch+1), "percentQoS_test_cost=", "{:.4f}".format(c_test_qos), "%_predict_test=", "{:.4f}".format(np.sum(cp_test) / len(test_points[:, 0])))

            # vCost_tr[epoch] = avg_cost
            # vCost_test[epoch] = c_test
            # vPred_test[epoch] = regu
    print("Optimization Finished!")


    cp, pv = sess.run([correct_prediction, predict_val], feed_dict={x: test_points, y: np.expand_dims(test_qos, axis=1)})
    print('Accuracy')
    print(np.sum(cp)/len(cp))

    print('False positive ratio (error type I)')
    pos = test_qos == 1
    pos_pred = pv[pos]
    type_I = np.sum(pos_pred == 0) / np.sum(pos)
    print(type_I)


    print('False negative ratio (error type II)')
    neg = test_qos == 0
    neg_pred = pv[neg]
    type_II = np.sum(neg_pred == 1) / np.sum(neg)
    print(type_II)

    test_pred = sess.run([pred_con], feed_dict={x: test_points})

    rmse = np.sqrt(np.sum((test_pred - np.expand_dims(test_con, axis=1))**2) / len(test_con))
    print('rmse = '+str(rmse))

    mae = np.sum(np.abs(test_pred - np.expand_dims(test_con, axis=1))) / len(test_con)
    print('mae = '+str(mae))



    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_path = saver.save(sess, model_path+'models_con_qos'+str(thres)+'_'+time.strftime("%y%m%d%H%M")+'.ckpt')
    print("Model saved in file: %s" % save_path)

    # training_vals = dict(vCost_tr=vCost_tr, vCost_test=vCost_test, vPred_tr=vPred_tr, vPred_test=vPred_test, nEpoch=training_epochs, learning_rate=con_learning_rate, batch_size=batch_size, n_hidden=con_n_hidden_1)
    # pickle.dump(training_vals, open(model_path+time.strftime("%y%m%d%H%M")+'_train_consumption'+'.p', "wb"))
