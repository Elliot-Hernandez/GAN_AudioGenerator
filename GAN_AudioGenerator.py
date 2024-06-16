import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import models
import librosa
import soundfile as sf
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm, trange

plt.style.use('seaborn-v0_8-whitegrid')
tf.random.set_seed(42)
frame_length =  1024
frame_step = 128
epochs = 3500
batch_size = 32
latent_dim = 100
SR_output = 16_000
d_loss, d_acc, gen_loss = [], [], []

datasetAudio = tf.keras.utils.audio_dataset_from_directory(
    '/Path/To/Sounds/',
    labels = None,
    label_mode = None,
    #sampling_rate = SR_output,
    batch_size = batch_size,
    shuffle = True,
    output_sequence_length = SR_output
)
print("datasetAudio")
print(datasetAudio)


def squeeze(audio):
  audio = tf.squeeze(audio, axis =- 1)
  return audio
datasetAudio = datasetAudio.map(squeeze, tf.data.AUTOTUNE)

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length = frame_length, frame_step = frame_step)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func = lambda audio: (get_spectrogram(audio)),
      num_parallel_calls = tf.data.AUTOTUNE)

datasetSpectrums = make_spec_ds(datasetAudio)
print("Spectrums")
print(datasetSpectrums)

for example_spectrograms in datasetSpectrums.take(1):
  break

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)


generator = tf.keras.Sequential([
    layers.Dense(256, input_shape = (latent_dim,), activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'),
    layers.BatchNormalization(),
    #layers.Dense(784, activation = 'tanh'),
    layers.Dense(118 * 513, activation = 'tanh'),
    layers.Reshape((118, 513, 1))
])
generator.summary()


discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape = (118, 513, 1)),#337*513
    layers.Dense(512, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid'),
    #layers.Dense(1, activation = 'sigmoid'),
])
discriminator.summary()


discriminator.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002),
                      loss = tf.keras.losses.BinaryCrossentropy(),
                      metrics = ['accuracy'])


discriminator.trainable = False
gan_input = tf.keras.Input(shape = (latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002),
            loss = tf.keras.losses.BinaryCrossentropy())


for epoch in range(epochs):
  print(f" {' ':^100s} Epoch {epoch + 1} / {epochs}")
  for real_sounds in datasetSpectrums:
    try:
      # Train discriminator
      random_latent_vectors = tf.random.normal(shape = (batch_size, latent_dim))
      generated_sounds = generator.predict(random_latent_vectors)
      combined_sounds = tf.concat([real_sounds, generated_sounds], axis = 0)
      labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis = 0)
      discriminator_loss = discriminator.train_on_batch(combined_sounds, labels)
      d_loss.append(discriminator_loss[0])
      random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
      misleading_labels = tf.ones((batch_size, 1))
      generator_loss = gan.train_on_batch(random_latent_vectors, misleading_labels)
      gen_loss.append(generator_loss)
    except ValueError:
      print('***********\n Error \n***********')

    # Print progress
  print(f"Discriminator Loss: {discriminator_loss[0]} | Discriminator Accuracy: {discriminator_loss[1]}")
  print(f"Generator Loss: {generator_loss}")

  plt.plot(d_loss, color='blue')
  plt.plot(gen_loss, color='red')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['d_loss', 'g_loss'], loc='upper left')

    # Generate and save samples
  if epoch % 100 == 0:
    print("Audio_Generado!")
    random_latent_vectors = tf.random.normal(shape = (1, latent_dim))
    generated_spectrum = generator.predict(random_latent_vectors)
    generated_spectrum = tf.squeeze(generated_spectrum, axis = [0,3])
    generated_spectrum = tf.make_tensor_proto(generated_spectrum)
    generated_spectrum = tf.make_ndarray(generated_spectrum)
    print(generated_spectrum.shape)
    audio_signal = librosa.griffinlim(generated_spectrum)
    sf.write(f"/Path/To/Folder/audio-{epoch}.wav", audio_signal, SR_output, 'PCM_24')


plt.show()
generator.save('/Path/To/Folder/GAN_AudioGenerator.h5')
#gan.save('/Path/To/Folder/GAN_1.h5')
#gan.save('/Path/To/Folder/GAN_1.keras')
#generator.save('/Path/To/Folder/GAN_1.keras')
for i in tqdm(range(epochs), token='token', chat_id='id_chat', ascii=True, desc='Finalizado'):
    pass