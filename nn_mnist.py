import gzip
import _pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print(train_y[57])


# TODO: the neural net!!

train_y =one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 25)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(25)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(25, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20

epoch = 0
perdidaAnt = 999999999
errores = []
erroresValid = []
while(True):
    for jj in range (int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    perdidaAct = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    errores.append(perdidaAct)
    perdidasValid = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    erroresValid.append(perdidasValid)
    print("Train #:", epoch, "Error: ", perdidaAct)
    print("Valid #:", epoch, "Error: ", perdidasValid)
    if (abs(perdidaAct - perdidaAnt)/perdidaAnt)<0.001:
        break;
    perdidaAnt = perdidaAct
    epoch += 1
    result = sess.run(y, feed_dict={x: batch_xs})
    #for b, r in zip(batch_ys, result):
    #    print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
contador = 0
inter = 0
print("Test")
resultado = sess.run(y, feed_dict={x: test_x})
for correcto, tested in zip(test_y, resultado):
    inter = inter+1
    #print(correcto, "-->", tested)
    if (np.argmax(correcto) == np.argmax(tested)):
        contador+=1
print("----------------------------------------------------------------------------------")
print("Porcentaje acierto = ")
print(contador/inter)

plt.subplot(1,2,1)
plt.plot(errores)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error de entrenamiento")

plt.subplot(1,2,2)
plt.plot(erroresValid)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error de validación")

plt.savefig("figura1.png")
plt.show()

