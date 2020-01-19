import numpy as np

from sparse_ae.model import create_model, load_encoder_decoder_pair
from sparse_ae.utils import plot_latent_unit_activation

image_shape = (28, 28)
num_classes = 10
input_size = 784
latent_size = 64
output_size = 784
sample_size = 6

autoencoder = create_model(input_size, latent_size, activity_regularizer=None)
_, decoder = load_encoder_decoder_pair(autoencoder, 'models/ae_l1.h5')

if __name__ == '__main__':
    """
    Сохраняет рекоструированные изображения по одному на каждый активный нейрон.  
    """
    latent_codes = np.eye(64)
    plot_latent_unit_activation(decoder, latent_codes, 'images/generated_images_l1.png', image_shape)
