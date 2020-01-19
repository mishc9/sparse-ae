import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.backend as K


def create_model(input_size: int,
                 latent_size: int,
                 activity_regularizer: Regularizer) -> Model:
    """
    Функция для создания автокодировщика.
    :param input_size: размерность входного вектора.
    :param latent_size: размерность латентного пространства.
    :param activity_regularizer: регуляризатор активации нейронов латентного слоя.
    :return: tf.keras.Model
    """
    output_size = input_size
    x = Input(shape=(input_size,), name='input')
    z = Dense(latent_size,
              activation='sigmoid',
              activity_regularizer=activity_regularizer,
              name='encoder')(x)
    r = Dense(output_size, activation='sigmoid', name='decoder')(z)
    return Model(x, r)


def load_encoder_decoder_pair(autoencoder: Model, path_to_weights: str) -> (Model, Model):
    """
    Создает из автокодировщика энкодер и декодер, сохраняя веса модели.
    :param autoencoder: автокодировщик
    :param path_to_weights: путь до весов модели
    :return: пара (энкодер, декодер)
    """
    autoencoder.load_weights(path_to_weights)

    encoder_layer = autoencoder.get_layer('encoder')
    decoder_layer = autoencoder.get_layer('decoder')
    input_tensor = autoencoder.get_layer('input').input

    latent_input = Input(shape=encoder_layer.output_shape[-1], name='latent_input')

    z = encoder_layer(input_tensor)
    reconstructed = decoder_layer(latent_input)

    decoder = Model(latent_input, reconstructed)
    encoder = Model(input_tensor, z)
    return encoder, decoder


# For nightly build.
# More: https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer
# @tf.keras.utils.register_keras_serializable(package='Custom', name='KL')
class KLRegularizer(Regularizer):
    """
    KL - регуляризатор
    """
    def __init__(self, rho, size_average=True):
        """
        Инициализация регуляризатора с заданным ро.
        :param rho: целевая вероятность активации нейрона.
        :param size_average: усреднение лосса
        """
        self.size_average = size_average
        self.rho = rho

    def __call__(self, data_rho):
        """
        D_KL(P||Q) = sum(p*log(p/q)) = -sum(p*log(q/p)) = -p*log(q/p) - (1-p)log((1-q)/(1-p))
        """
        eps = K.epsilon()  # ближайшее к нулю значение типа float
        dkl = (- self.rho       * tf.math.log(eps + data_rho       / (self.rho + eps))
               - (1 - self.rho) * tf.math.log(eps + (1 - data_rho) / (1 - self.rho + eps)))
        if self.size_average:
            _rho_loss = tf.reduce_mean(dkl)
        else:
            _rho_loss = tf.reduce_sum(dkl)
        return _rho_loss

    def get_config(self):
        return {'rho': float(self.rho), 'size_average': self.size_average}


def make_encoder(model: Model) -> Model:
    """
    "Извлечение" энкодера из автокодировщика.
    :param model: автокодировщик
    :return: извлеченный энкодер
    """
    input = model.get_layer('input')
    encoder = model.get_layer('encoder')
    return Model(input.input, encoder.output)