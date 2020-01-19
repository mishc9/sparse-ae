import matplotlib.pyplot as plt
import tensorflow as tf

from sparse_ae.model import KLRegularizer, create_model, make_encoder
from sparse_ae.utils import load_mnist

image_shape = (28, 28)
num_classes = 10
input_size = 784
latent_size = 64
output_size = 784
sample_size = 6

if __name__ == '__main__':
    """
    Сравнение распределений активации нейронов в латентном слое.
    """
    model_kl = create_model(input_size, latent_size, KLRegularizer(0.1))
    model_kl.load_weights('models/ae_kl.h5')
    model_vanilla = tf.keras.models.load_model('models/ae_vanilla.h5')
    model_l1 = tf.keras.models.load_model('models/ae_la.h5')

    models = {'vanilla': model_vanilla, 'l1': model_l1, 'kl': model_kl}
    models = {k: make_encoder(m) for k, m in models.items()}

    (X_train, y_train), (X_test, y_test) = load_mnist(input_size)

    for directory, model in models.items():
        values = model.predict(X_test)
        values = values.mean(axis=0)
        plt.hist(values, bins=25, alpha=0.5,
                 label=directory, stacked=True,
                 density=True, )

    plt.legend(loc='upper right')
    plt.savefig('images/hidden_activations.png')
