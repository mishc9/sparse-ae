import os
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from sparse_ae.model import create_model
from sparse_ae.utils import load_mnist, get_sample, train_val_split, plot_reconstruction

images = 'images'
models = 'models'
os.makedirs(images, exist_ok=True)
os.makedirs(models, exist_ok=True)

save_to = os.path.join(images, 'reconstruction_vanilla.png')
model_path = os.path.join(models, 'ae_vanilla.h5')
log_dir = 'tb_logs/vanilla'
image_shape = (28, 28)
num_classes = 10
input_size = 784
latent_size = 64
output_size = 784
sample_size = 6

if __name__ == '__main__':
    """
    Создание и обучение автоэкодера без регуляризации.
    """
    autoencoder = create_model(input_size, latent_size, activity_regularizer=None)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    (X_train, y_train), (X_test, y_test) = load_mnist(input_size)
    X, X_val = train_val_split(X_train, 0.8)

    autoencoder.fit(X, X,
                    validation_data=(X_val, X_val),
                    epochs=200,
                    batch_size=64,
                    callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=1),
                               ModelCheckpoint(model_path, str='val_loss', save_best_only=True),
                               EarlyStopping(patience=5, min_delta=0.0001),
                               ]
                    )

    sample = get_sample(X_test, sample_size)
    plot_reconstruction(autoencoder, sample, save_to, (28, 28))
