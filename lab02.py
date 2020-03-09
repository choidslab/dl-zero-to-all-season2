import tensorflow as tf
import numpy as np

# print(tf.__version__)

x_data = [1, 2, 3, 4, 5]  # input data
y_data = [1, 2, 3, 4, 5]  # output data

W = tf.Variable(2.9)  # initialize Weight
b = tf.Variable(0.5)  # initialize bias

learning_rate = 0.01  # learning rate --> weight, bias 업데이트 비율을 조정하는 값

for i in range(100):  # Update W, b
    # Gradient Descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:
        print("{:5} | {:10.4f} | {:10.4f} | {:10.6f}".format(i, W.numpy(), b.numpy(), cost))
print()

print(W * 5 + b)  # Predict
print(W * 2.5 + b)
