import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Genarator(keras.Model):

    def __init__(self):
        super(Genarator, self).__init__()
    # z:[b,100]=>[b,64,64,3]
        self.fc = layers.Dense(3*3*512)
        self.cov1 = layers.Convolution2DTranspose(256,3,3,'valid')
        self.bn1 = layers.BatchNormalization()

        self.cov2 = layers.Convolution2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.cov3 = layers.Convolution2DTranspose(3, 4, 3, 'valid')
        self.bn3 = layers.BatchNormalization()



    def __call__(self, inputs, training=None):

        x = self.fc(inputs)
        x = tf.reshape(x,[-1,3,3,512])

        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.cov1(x),training=training))
        x = tf.nn.leaky_relu(self.bn1(self.cov1(x), training=training))
        x = self.bn3(self.cov3(x),training=training)
        x = tf.nn.tanh(x)
        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64,5,3,'valid')
        self.conv2 = layers.Conv2D(128,5,3,'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256,5,3,'valid')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1,activation='sigmoid')

    def __call__(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(inputs),training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(inputs), training=training))
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
def main():
    d = Discriminator()
    g = Genarator()

    x = tf.random.normal([2,64,64,3])
    z = tf.random.normal([2,100])

    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)


if __name__ == '__main__':
    main()