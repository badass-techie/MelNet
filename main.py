import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import Input, Model, layers, optimizers, backend as K
import numpy as np
from matplotlib import pyplot as plt

import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display

import os
import time
from tqdm import tqdm



### constants
NUM_LAYERS = 10
MIXTURE_NUM_FILTERS = 10
F_MAX = None
T_MAX = None
N_MELS = 256
N_FFT = 2048
# HOP_LENGTH = 512
SAMPLING_RATE = 12000
CLIP_LENGTH = 20
AUDIO_LENGTH = 14164 - CLIP_LENGTH
DATASET_SIZE = 1000
DATASET = None


### helpers ###

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))

def load_dataset(filename="dataset/Schubert Piano Music.wav"):
    global DATASET, F_MAX, T_MAX
    indices = np.random.choice(AUDIO_LENGTH, size=DATASET_SIZE, replace=False)
    spectrograms = []

    print("\nLoading dataset into memory...\n")
    for i, index in enumerate(tqdm(indices)):
        time_series, _ = librosa.load(filename, sr=SAMPLING_RATE, mono=True, offset=index, duration=CLIP_LENGTH)
        spectrogram = librosa.feature.melspectrogram(y=time_series, sr=SAMPLING_RATE, n_mels=N_MELS, n_fft=N_FFT)
        spectrograms.append(np.expand_dims(spectrogram, axis=0))
        F_MAX = spectrogram.shape[0]
        T_MAX = spectrogram.shape[1]
        # print("\nF_MAX: {}\nT_MAX: {}\n".format(F_MAX, T_MAX))

    DATASET = np.concatenate(spectrograms, axis=0)

def get_batch(batch_size=16):
    indices = np.random.choice(DATASET_SIZE, size=batch_size, replace=False)
    return DATASET[indices]


### model ###

def time_delayed_stack(delayed_time_spectrogram):
    # time axis
    row_1 = tf.expand_dims(delayed_time_spectrogram[:, 0, 0:T_MAX], axis=-1) # row of spectrogram sliced to length of current time step
                                                                         # extra axis added for features, to change shape from (batch_size, seq_len) to (batch_size, seq_len, feature_len)
    time_rnn = layers.LSTM(NUM_LAYERS, input_shape=(None, T_MAX, 1), return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0, name='time_rnn_from_time_delayed_stack')
    time_rnn_out, _, _ = time_rnn(row_1)
    time_rnn_out = tf.expand_dims(time_rnn_out, axis=1) # add axis for concatenation with more rows
    for i in range(1, F_MAX):
        row_i = tf.expand_dims(delayed_time_spectrogram[:, i, 0:T_MAX], axis=-1)
        row_i_out, _, _ = time_rnn(row_i)
        row_i_out = tf.expand_dims(row_i_out, axis=1)
        time_rnn_out = layers.Concatenate(axis=1)([time_rnn_out, row_i_out])

    # freq axis
    col_1 = tf.expand_dims(delayed_time_spectrogram[:, :, 0], axis=-1) # freq axis flipped to match indexing order of matrix
                                                                                   # extra axis added for features, to change shape from (batch_size, seq_len) to (batch_size, seq_len, feature_len)
    freq_rnn = layers.LSTM(NUM_LAYERS,  input_shape=(None, delayed_time_spectrogram.shape[1], 1), return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0, name='freq_rnn_from_time_delayed_stack')
    freq_rnn_out, _, _ = freq_rnn(col_1)
    freq_rnn_out = tf.expand_dims(freq_rnn_out, axis=2) # add axis for concatenation with more columns
    for i in range(1, T_MAX):
        col_i = tf.expand_dims(delayed_time_spectrogram[:, :, i], axis=-1)
        col_i_out, _, _ = freq_rnn(col_i)
        col_i_out = tf.expand_dims(col_i_out, axis=2)
        freq_rnn_out = layers.Concatenate(axis=2)([freq_rnn_out, col_i_out])

    # inverted freq axis
    col_1 = tf.expand_dims(delayed_time_spectrogram[:, :, 0], axis=-1)  # extra axis added for features, to change shape from (batch_size, seq_len) to (batch_size, seq_len, feature_len)
    reverse_freq_rnn = layers.LSTM(NUM_LAYERS, input_shape=(None, delayed_time_spectrogram.shape[1], 1), return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0, name='reverse_freq_rnn_from_time_delayed_stack')
    reverse_freq_rnn_out, _, _ = reverse_freq_rnn(col_1)
    reverse_freq_rnn_out = tf.expand_dims(reverse_freq_rnn_out, axis=2)  # add axis for concatenation with more columns
    for i in range(1, T_MAX):
        col_i = tf.expand_dims(delayed_time_spectrogram[:, :, i], axis=-1)
        col_i_out, _, _ = freq_rnn(col_i)
        col_i_out = tf.expand_dims(col_i_out, axis=2)
        reverse_freq_rnn_out = layers.Concatenate(axis=2)([reverse_freq_rnn_out, col_i_out])

    result = (tf.concat([time_rnn_out, freq_rnn_out, reverse_freq_rnn_out], axis=3))    # concatenate the 3 feature maps
    return result

def freq_delayed_stack(delayed_freq_spectrogram, time_delayed_stack_out):
    time_input = tf.transpose(time_delayed_stack_out, [0, 1, 3, 2])
    row_1 = tf.expand_dims(delayed_freq_spectrogram[:, 0:F_MAX, 0], axis=2)
    freq_rnn = layers.LSTM(1, input_shape=(None, F_MAX, 1), return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0, name='freq_rnn_from_freq_delayed_stack')
    freq_rnn_out, _, _ = freq_rnn(row_1)
    for i in range(1, T_MAX):
        row_i = tf.expand_dims(delayed_freq_spectrogram[:, 0:F_MAX, i], axis=2)
        row_i_out, _, _ = freq_rnn(row_i)

        # apply output of time delayed stack
        for j in range(NUM_LAYERS):
            W = tf.Variable(shape=[F_MAX, 1], dtype=tf.float32, initial_value=np.ones((F_MAX, 1)) * 0.1)
            row_i_out += freq_rnn(W * row_i_out + tf.expand_dims(time_input[:, j, 0:F_MAX, i], axis=2))[0]
        freq_rnn_out = tf.concat([freq_rnn_out, row_i_out], axis=2)

    return freq_rnn_out

def build_model():
    spectrogram = Input(shape=(F_MAX, T_MAX))  # actual shape is (batch_dim, F_MAX, T_MAX)
    delayed_freq = tf.roll(spectrogram, shift=1, axis=1)    # shift freq axis down by one
    delayed_time = tf.roll(spectrogram, shift=1, axis=2)    # shift time axis to the left by one

    time_delayed_stack_out = time_delayed_stack(delayed_time)
    time_delayed_stack_out = layers.BatchNormalization()(time_delayed_stack_out)

    W = tf.Variable(shape=[3 * NUM_LAYERS, NUM_LAYERS], dtype=tf.float32, initial_value=tf.ones(shape=[3 * NUM_LAYERS, NUM_LAYERS]))
    W = tf.expand_dims(W, axis=0)    # add dummy batch dimension
    weighted_out = layers.dot([time_delayed_stack_out, W], axes=[-1, 1])
    weighted_out = tf.transpose(weighted_out, [0, 3, 2, 1])

   # residual/skip connections with output from time delayed stack
    residual_out = tf.expand_dims(tf.transpose(delayed_time, [0, 2, 1]), axis=1)
    for i in range(0, NUM_LAYERS - 1, 1):
        residual_out = layers.Concatenate(axis=1)([residual_out, tf.expand_dims(weighted_out[:, i, :, :], axis=1)])

    to_freq_delayed_stack= weighted_out + residual_out
    freq_delayed_stack_out = freq_delayed_stack(delayed_freq, to_freq_delayed_stack)
    freq_delayed_stack_out = layers.BatchNormalization()(freq_delayed_stack_out)

    # µ, σ, α to sample from
    # µ
    mu = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(tf.expand_dims(freq_delayed_stack_out, axis=3))
    mu = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(mu)
    mu = layers.BatchNormalization()(mu)

    # σ
    sigma = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(tf.expand_dims(freq_delayed_stack_out, axis=3))
    sigma = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(sigma)
    sigma = layers.BatchNormalization()(sigma)
    sigma = K.exp(sigma)

    # α
    alpha = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(tf.expand_dims(freq_delayed_stack_out, axis=3))
    alpha = layers.Conv2D(filters=MIXTURE_NUM_FILTERS, kernel_size=(1, 1), padding='same', activation='relu')(alpha)
    alpha = layers.BatchNormalization()(alpha)
    alpha = K.exp(alpha)

    # make sum of α one
    norm_alpha = tf.expand_dims(alpha[:, :, :, 0] / K.sum(alpha, axis=3), axis=3)
    for i in range(1, MIXTURE_NUM_FILTERS):
        norm_alpha = tf.concat([norm_alpha, tf.expand_dims(alpha[:, :, :, i] / K.sum(alpha, axis=3), axis=3)], axis=3)

    out = layers.Concatenate(axis=1)([tf.expand_dims(norm_alpha, axis=1), tf.expand_dims(mu, axis=1), tf.expand_dims(sigma, axis=1)])
    return Model(spectrogram, out)


### training ###

@tf.function
def train_step(model, optimizer, labels):
    with tf.GradientTape() as tape:
        logits = model(labels)  #labels same as inputs
        mu = logits[:, 1, :]
        sigma = logits[:, 2, :]
        alpha = logits[:, 0, :]

        mixture = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                                        components_distribution=tfd.Normal(loc=mu, scale=sigma))
        log_loss = -tf.reduce_sum(mixture.log_prob(tf.squeeze(labels)))

    gradients = tape.gradient(log_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return log_loss

def train(model, batch_size=16, num_iterations=int(1e5), steps=1000, lr=7e-4):
    """training loop (num_iterations has to be a multiple of steps or it will be truncated)"""

    optimizer = optimizers.RMSprop(learning_rate=lr, momentum=0.9)
    loss_history = []
    prev_time = time.time()
    time_elapsed = 0

    # load saved models or create if they don't exist
    if os.path.isfile("models/weights.h5"):
        model.load_weights("models/weights.h5")
    else:
        model.save_weights("models/weights.h5")

    print("Training...")

    for i in range(0, num_iterations, steps):
        for j in tqdm(range(steps)):
            batch = get_batch(batch_size)
            loss = train_step(model, optimizer, batch)
            loss_history.append(loss.numpy().mean())

            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print(f"Iteration {i + steps}/{num_iterations}. Loss: {loss_history[-1]}. Time elapsed: {format_time(time_elapsed)}\n")
        # save checkpoints
        model.save_weights("models/weights.h5")
        model.save_weights(f"models/weights{i + steps}.h5")

        # plot a graph that will show how our loss varied with time
        plt.plot(loss_history)
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join("./plots", "Training Progress"))
        # plt.show()
        plt.close()

        # evaluate
        generate(model, get_batch(1), [f"{i + steps}"])

        if (i + steps) % 10000 == 0:
            lr /= 2 # half learning rate


### inference ###

# samples the distribution learned by the model to create a spectrogram
def sample(logits):
    mu = logits[:, 1, :, :, :]
    sigma = logits[:, 2, :, :, :] * 0.5
    alpha = logits[:, 0, :, :, :]

    batch_size = logits.shape[0]
    spectrogram = np.zeros((1, F_MAX, T_MAX))   # first dimension represents batch size

    for batch_num in range(batch_size):
        alpha_batch = alpha[batch_num]
        mu_batch = mu[batch_num]
        sigma_batch = sigma[batch_num]

        # make alpha add up to one
        sum_alpha_batch = np.sum(alpha_batch, axis=2)
        for l in range(MIXTURE_NUM_FILTERS):
            alpha_batch[:, :, l] /= sum_alpha_batch

        out = np.zeros((F_MAX, T_MAX))
        for f in range(F_MAX):
            for t in range(T_MAX):
                indices = np.random.choice(np.arange(0, MIXTURE_NUM_FILTERS), p=np.ravel([alpha_batch[f, t, :]]))
                out[f, t] = np.random.normal(mu_batch[f, t, indices], 0.221 * sigma_batch[f, t, indices])

        spectrogram = np.append(spectrogram, np.expand_dims(out, axis=0), axis=0)
    return spectrogram

def generate(model, batch, output_names, prefix_len=5):
    batch = (batch - np.min(batch)) / np.max(batch) #normalize

    for i, spectrogram in enumerate(batch):
        spectrogram[:, prefix_len:] = 0
        for col in range(prefix_len, spectrogram.shape[1]):
            for row in range(spectrogram.shape[0]):
                params = model(np.expand_dims(spectrogram, 0))
                result = sample(params)[0]
                spectrogram[row, col] = result[row, col]

        # reconstruct signal and save as wav file
        time_series = librosa.feature.inverse.mel_to_audio(spectrogram)
        sf.write(f"saved/{output_names[i]}.wav", time_series, SAMPLING_RATE)

        # save an image of the spectrogram
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(spectrogram, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SAMPLING_RATE, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title=output_names[i])
        plt.savefig(f"saved/{output_names[i]}.png")



if __name__ == '__main__':
    load_dataset()
    model = build_model()
    train(model)
