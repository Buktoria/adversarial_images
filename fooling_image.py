import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,  name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
b_conv1 = bias_variable([32], name='b_conv1')

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
b_conv2 = bias_variable([64], name='b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
b_fc1 = bias_variable([1024], name='b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10], name='W_fc2')
b_fc2 = bias_variable([10], name='b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
grad = tf.gradients(-cross_entropy, x)

saver = tf.train.Saver()

# Create adversarial image for digit 6 with 2
with tf.Session() as sess:
	# Restore variables from disk.
	saver.restore(sess, "./tmp/model.ckpt")

	print('Done Restoring')

	# Get digit 2 images
	digit_2_images = [i for i, l in zip(mnist.test.images, mnist.test.labels) if l[2] == 1]
	digit_6_images = [i for i, l in zip(mnist.test.images, mnist.test.labels) if l[6] == 1]
	
	digit_2_label = [0, 0, 1, 0, 0, 0, 0 ,0, 0, 0]
	digit_5_label = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 
	digit_6_label = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	digit_7_label = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 

	
	# Create Adversary Image

	# Get image gradiant with the class we want it to classify as
	image_gradiant = sess.run(
		grad, 
		feed_dict={x:[digit_2_images[10]], y_:[digit_6_label], keep_prob:1.0}
	)[0][0]

	print(image_gradiant)

	# image_gradiant = sess.run(
	# 	grad, 
	# 	feed_dict={x:[digit_2_images[10]], y_:[digit_5_label], keep_prob:1.0}
	# )[0][0]

	# print(image_gradiant)

	# We want to know if we need to incress or decress a pixel of the image
	image_pixel_direction = np.sign(image_gradiant)

	# Perform an image update

	# my_classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [new_image]})

	# print(digit_2_images[10])
	# print(image_gradiant)
	# print(image_pixel_direction)


	pred2 = sess.run(y_conv, feed_dict={x:[digit_2_images[10]], keep_prob:1.0})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)


	new_image = digit_2_images[10] + image_pixel_direction * 0.001 
	pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.0})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)

	_, (ax1, ax2, ax3), = plt.subplots(1, 3)
	ax1.imshow(digit_2_images[10].reshape(28, 28), cmap=plt.cm.Greys);
	ax2.imshow(np.array(image_gradiant).reshape(28,28), cmap=plt.cm.Greys);
	ax3.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);
	plt.show()


	new_image = digit_2_images[10] + image_pixel_direction * 0.01 
	pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.0})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)

	_, (ax1, ax2, ax3), = plt.subplots(1, 3)
	ax1.imshow(digit_2_images[10].reshape(28, 28), cmap=plt.cm.Greys);
	ax2.imshow(np.array(image_gradiant).reshape(28,28), cmap=plt.cm.Greys);
	ax3.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);
	plt.show()

##############

	new_image = digit_2_images[10] + image_pixel_direction * 0.1
	pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.0})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)

	_, (ax1, ax2, ax3), = plt.subplots(1, 3)
	ax1.imshow(digit_2_images[10].reshape(28, 28), cmap=plt.cm.Greys);
	ax2.imshow(np.array(image_gradiant).reshape(28,28), cmap=plt.cm.Greys);
	ax3.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);
	plt.show()

	new_image = digit_2_images[10] + image_pixel_direction 
	pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.0})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)

	_, (ax1, ax2, ax3), = plt.subplots(1, 3)
	ax1.imshow(digit_2_images[10].reshape(28, 28), cmap=plt.cm.Greys);
	ax2.imshow(np.array(image_gradiant).reshape(28,28), cmap=plt.cm.Greys);
	ax3.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);
	plt.show()

	new_image = digit_2_images[10] + image_pixel_direction 
	pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.5})
	label2 = np.argmax(pred2)
	print(label2)
	print(pred2)


	# Show images
	_, (ax1, ax2, ax3), = plt.subplots(1, 3)
	ax1.imshow(digit_2_images[10].reshape(28, 28), cmap=plt.cm.Greys);
	ax2.imshow(np.array(image_gradiant).reshape(28,28), cmap=plt.cm.Greys);
	ax3.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);
	plt.show()

	# NOTE TO SELF< WE ARE GETTING NEGTIVE PROP. 
	# WHICH MEANS WE SHOULD BE TAKING THE SOFT MAX


