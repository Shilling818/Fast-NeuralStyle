Tensorflow version for the paper 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution' 

In the 'net.py', you may use vgg16_tf.mat. You can download the standard VGG-16 weights and biases, and then compress them into a mat using Matlab. 

My propose is to transform a optical image into a SAR image. These two type images can be considered as two kinds of style.
I use a SAR image as a style image. 14 optical images as content images. 1 optical image as a test content image. The image size is 700*700

The origin optical image is ![image](http://github.com/Shilling818/Fast-NeuralStyle/raw/master/image/origin.png)

The target SAR image is ![image](http://github.com/Shilling818/Fast-NeuralStyle/raw/master/image/target.png)

The transformed SAR image is ![image](http://github.com/Shilling818/Fast-NeuralStyle/raw/master/image/reality.png)