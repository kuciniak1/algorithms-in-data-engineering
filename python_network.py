from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import time
import tracemalloc


def train_network():
    start = time.time()
    model.fit(train_images, train_labels, epochs=3, batch_size=100)
    end = time.time()
    print(end - start)

def calculate_memory_allocated(snapshot1, snapshot2):
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in stats)
    return total_memory

if __name__ == "__main__":

    model = Sequential()

    model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(28, 28, 1), use_bias=True))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', use_bias=True))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(84, activation='relu', use_bias=True))
    model.add(Dense(10, activation='softmax', use_bias=True))


    sgd_optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    train_network()

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    total_memory_allocated = calculate_memory_allocated(snapshot1, snapshot2)
    print(f'Total memory allocated: {total_memory_allocated / 1024:.2f} KB')

    start = time.time()
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    end = time.time()
    print(end - start)
    print(f'Test accuracy: {test_accuracy}')