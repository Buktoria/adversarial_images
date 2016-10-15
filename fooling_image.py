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
	
	
	# Create Adversary Image
	digit_6_label = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	alpha = float(input('Value for alpha = ')) # tested with value 0.2


	# Perform an image update for 10 images
	total = 0
	miss_classified = 0
	for image in digit_2_images:
		
		# After doing this to 10 images stop
		if total == 10:
			break

		# Only create adversary images for those that are oringally 
		# correctly classified
		if np.argmax(
			sess.run(y_conv, feed_dict={x:[image], keep_prob:1.0})
		) == 2:
			

			# Get image gradiant with the class we want it to classify as
			delta = sess.run(
				grad, 
				feed_dict={x:[image], y_:[digit_6_label], keep_prob:1.0}
			)[0][0]

			# We want to know if we need to incress 
			# or decress a pixel of the image
			image_pixel_direction = np.sign(delta)
			
			new_image = image + image_pixel_direction * alpha
			pred2 = sess.run(y_conv, feed_dict={x:[new_image], keep_prob:1.0})
			label = np.argmax(pred2)

			# Only show correctly 6 classified images
			if label == 6:
	
				plt.subplot(10, 3, (total*3 + 1))
				# plt.title('Original')
				plt.imshow(image.reshape(28, 28), cmap=plt.cm.Greys);
				plt.subplot(10, 3, (total*3 + 2))
				# plt.title('Delta')
				plt.imshow(np.array(delta).reshape(28,28), cmap=plt.cm.Greys);
				plt.subplot(10, 3, (total*3 + 3))
				# 
				plt.imshow(new_image.reshape(28,28), cmap=plt.cm.Greys);


				total += 1
			
			else:
				miss_classified += 1

# Output Stat.
print('Alpha = {}'.format(alpha))
print('Total Miss Classified = {}'.format(miss_classified))
print('Perentage Right When Modified = {}'.format(10/(miss_classified+10)))

# Display Results
plt.subplot(10, 3, 1)
plt.title('Original')
plt.subplot(10, 3, 2)
plt.title('Delta')
plt.subplot(10, 3, 3)
plt.title('New Image - Classification 6')

plt.show()



