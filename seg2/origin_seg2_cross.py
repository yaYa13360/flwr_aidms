import tensorflow_datasets as tfds
import timeit
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold, cross_val_score

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')
#         self.samples = tf.Variable(initial_value=0.0, trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_np = y_true.numpy()
        y_pred_np = create_mask(y_pred).numpy()
        smooth = 100
        dice = 0
        for task_id in list(np.unique(y_true)):
            y_true_task = (y_true_np == task_id).astype(np.float32)
            y_pred_task = (y_pred_np == task_id).astype(np.float32)
            intersection = np.sum(y_true_task * y_pred_task)
            union = np.sum(y_true_task) + np.sum(y_pred_task)
            dice += (2. * intersection + smooth) / (union + smooth)
        dice = dice/len(np.unique(y_true))

        self.dice.assign_add(dice)
        self.samples.assign(self.samples+1)

    def result(self):
        return self.dice / self.samples

    def reset_state(self):
        self.dice.assign(0.0)
        self.samples.assign(0.0)

class DiceCoefficientCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(DiceCoefficientCallback, self).__init__()
        self.validation_data = validation_data
        self.dice_scores = []

    def on_epoch_end(self, epoch, logs=None):
        dice_metric = DiceCoefficient()
        for x_val, y_val in self.validation_data:
            predictions = self.model.predict(x_val)
            dice_metric.update_state(y_val, predictions)
        dice_score = dice_metric.result().numpy()
        self.dice_scores.append(dice_score)


(train, test), metadata = tfds.load(
    'oxford_iiit_pet',
    split=['train', 'test'],
    with_info=True,
    )

BATCH_SIZE = 64

test_images = test.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_images.batch(BATCH_SIZE)

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

OUTPUT_CLASSES = 3

# k-fold---------------------------------------------------------------
N_SPLITS = 3
data_size = len(train)
model_history_arr = []
test_accuracies = []

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(range(data_size))):
    train_dataset = train.skip(train_index[0]).take(len(train_index))
    val_dataset = train.skip(val_index[0]).take(len(val_index))
    train_fold_D_parallel = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_fold_D_parallel = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = (
        train_fold_D_parallel
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = val_fold_D_parallel.batch(BATCH_SIZE)
    
    # model
    model = unet_model(output_channels=OUTPUT_CLASSES)
    dice_coefficient = DiceCoefficient()
    model.compile(optimizer='adam',
            run_eagerly=True,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy',  {'dice': dice_coefficient}])
    
    # TRAIN_LENGTH = len(train)
    # VAL_SUBSPLITS = 5
    # STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE//VAL_SUBSPLITS
    
    dice_callback = DiceCoefficientCallback(validation_data=val_dataset)
    model_history = model.fit(train_dataset,
                              epochs=10,
                              validation_data=val_dataset,
                              callbacks=[dice_callback])
    model_history_arr.append(model_history)
    test_results  = model.evaluate(test_dataset)
    test_accuracies.append(test_results[1])
#     tf.keras.backend.clear_session()
#     train_dataset = None
#     val_dataset = None


# plot------------------------------------------------------------
color_arr = ['pink', 'blue', 'yellow', 'green', 'purple']

# loss
for i in  range(N_SPLITS):
    val_loss = model_history_arr[i].history['val_loss']
    loss = model_history_arr[i].history['loss']
    plt.plot(val_loss, linestyle='--', label = (str(i+1)+': val_loss'), color = color_arr[i])
    plt.plot(loss, linestyle='-', label = (str(i+1)+': loss'), color = color_arr[i])        
plt.legend(loc='upper right')
plt.title('loss curve')
plt.savefig("loss.png")
plt.clf()

# acc
for i in  range(N_SPLITS):
    val_accuracy = model_history_arr[i].history['val_accuracy']
    accuracy = model_history_arr[i].history['accuracy']
    plt.plot(val_accuracy, linestyle='--', label = (str(i+1)+': val_accuracy'), color = color_arr[i])
    plt.plot(accuracy, linestyle='-', label = (str(i+1)+': accuracy'), color = color_arr[i])
plt.legend(loc='lower right')
plt.title('accuracy curve')
plt.savefig("accuracy.png")
plt.clf()

# dice
for i in  range(N_SPLITS):
    val_dice_coefficient = model_history_arr[i].history['val_dice_coefficient']
    dice_coefficient= model_history_arr[i].history['dice_coefficient']
    plt.plot(val_dice_coefficient, linestyle='--', label = (str(i+1)+': val_dice_coefficient'), color = color_arr[i])
    plt.plot(dice_coefficient, linestyle='-', label = (str(i+1)+': dice_coefficient'), color = color_arr[i])
plt.legend(loc='lower right')
plt.title('dice curve')
plt.savefig("dice.png")
plt.clf()

print('########## test acc ##########')    
for i in range(len(test_accuracies)):
    print('test {}: {:.4f}'.format(i + 1, test_accuracies[i]))
