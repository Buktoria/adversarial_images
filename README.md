# Adversarial Images

Based on the following article, we can create images that a cnn will misclassify with high confidence by tweaking pixel values slightly.

http://karpathy.github.io/2015/03/30/breaking-convnets/

This is a simple implementation of that given simple cnn provided by a tutorial by Tensorflow.

https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts

## Requirements

Please use python 3.4 or higher.

[tenserflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)

[numpy ](http://www.scipy.org/scipylib/download.html)

## How to Run

#### Step One Create the Model

To build that model first run work.py. This repo comes with a serialized version of the model I build and based my results off of. If you would like to use this one, go head to the next part.

#### Step Two Create Adversarial Images

Run the fooling_images.py file. It will out the first 10 images it fools a 2 to thinking its a 6. You can pass in a value alpha when you start. This values will used to control the amount each pixel is tweaked.

After the script runs, it will output a misclassification score after image transformation. It will then plot the ten images it fooloed the classifier with.

## Intersting Insights

One of the interesting things from this experiment was the different results you would get when modify alpha. Alpha was used as a parameter to tweak how much pixel values would change. I first started out pick ones that where quite low because they visually did less disruption to the image. This resulted in a high number of miss classified results after the image was modified. Then looking back at the article I choose a higher one of 0.5. Though this worked a lot better I was unsatisfied with the visual results of the image. I found the best value was around 0.2 which yielded valid results for the new images about 0.35% of the time. 

The resulting images are a bit distorted. This is due to the fact that the images are of a low resolution but also because in out cnn model we are taking the mean of the softmax for the cross entropy. This means that the cross entropy has undergone regularization. This leads to higher distortion as mentioned in the article.
