# Adversarial Images

Based on the following article, we can create images that a cnn will missclassify with high confidance by tweeking pixel values slightly.

http://karpathy.github.io/2015/03/30/breaking-convnets/

This is a simple implamentation of that given simple cnn provided by a tutorial by Tensorflow.

https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#deep-mnist-for-experts

## Requirments

Please use python3.4 or higher.

[tenserflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)

[numpy ](http://www.scipy.org/scipylib/download.html)

## Intersting Insights

One of the intersting things from this exparament was the different results you would get when modify alpha. Alpha was used as a parameter to tweek how much pixel values would change. I first started out pick ones that where quite low because they visually did less disruption to the image. This resulted in a high number of miss classifed results after the image was modified. Then looking back at the article I choose a higher one of 0.5. Though this worked a lot better I was unstisfied with the visual results of the image. I found the best value was around 0.2 which yieled valid results for the new images about 65% of the time. 

The resulting images are a bit distoried. This is due to the fact that the images are of a low resalution but also because in out cnn model we are taking the mean of the softmax for the cross entroy. This means that the cross entroy has undergone regularization. This leads to higher distortion as mentioned in the article.