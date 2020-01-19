from sparse_ae.model import create_model, load_encoder_decoder_pair
from sparse_ae.utils import load_mnist, get_one_item_of_each_class, reconstruct_from_top_k, \
    plot_reconstructed_from_top_k

image_shape = (28, 28)
num_classes = 10
input_size = 784
latent_size = 64
output_size = 784
sample_size = 10

if __name__ == '__main__':
    """
    Скрипт строит графики с реконструкцией по k максимальным активациям латентного слоя.
    """
    autoencoder = create_model(input_size, latent_size, input_size, None)
    (X_train, y_train), (X_test, y_test) = load_mnist(input_size)
    sample = get_one_item_of_each_class(X_test, y_test)
    k_active_neurons = (4, 8, 16, 32, 64)

    for active_neurons in k_active_neurons:
        image_map = {'original': sample}
        for path in ('models/ae_l1.h5', 'models/ae_kl.h5', 'models/ae_vanilla.h5'):
            encoder, decoder = load_encoder_decoder_pair(autoencoder, path)
            reconstructed = reconstruct_from_top_k(encoder, decoder, active_neurons, sample)
            image_map[path] = reconstructed
        plot_reconstructed_from_top_k(image_map,
                                      f'images/neurons_active_{active_neurons}.png',
                                      (28, 28),
                                      active_neurons)
