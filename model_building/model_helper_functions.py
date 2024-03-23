import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import copy
import os
import shutil
from keras import layers
from collections import OrderedDict
from keras import metrics
from keras.models import Sequential
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
try:
    from .data_pipeline import create_tensorflow_datasets
except:
    from data_pipeline import create_tensorflow_datasets
from time import gmtime, time
import requests
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ['grizzly', 'black', 'teddy']

def get_augmentation_layer(flip=True, rotate_factor=0, zoom_factor=0, random_flip_str='horizontal', input_shape=None):
    """
    Creates the augmentation layer.

    Args:
        flip: Boolean for whether or not to randomly flip the image.
        rotate_factor: A float defining how much to randomly rotate the image.
        zoom_factor: A float defining how much to randomly zoom the image.
        random_flip_str: A string to pass into RandomFlip defining the flip method.
        input_shape: The input image shape.
    
    Returns:
        The tensorflow augmentation layer.
    """

    input_used = False

    data_augmentation = Sequential()


    if flip:
        data_augmentation.add(layers.RandomFlip(random_flip_str, input_shape=input_shape))
        input_used = True
    
    if rotate_factor:
        if input_used:
            data_augmentation.add(layers.RandomRotation(factor=rotate_factor, fill_mode='reflect'))
        else:
            data_augmentation.add(layers.RandomRotation(factor=rotate_factor, fill_mode='reflect', input_shape=input_shape))
            input_used = True

    if zoom_factor:
        if input_used:
            data_augmentation.add(layers.RandomZoom(height_factor=zoom_factor, width_factor=zoom_factor, fill_mode='reflect'))
        else:
            data_augmentation.add(layers.RandomZoom(height_factor=zoom_factor, width_factor=zoom_factor, fill_mode='reflect', input_shape=input_shape))
            input_used = True


    return data_augmentation


def build_model(image_shape, augmentation_layer, cnn_units, cnn_filters, cnn_strides, dropout, dense_units, activation, optimizer):
    """
    Builds the tensorflow model.

    Args:
        image_shape: The shape of the image.
        augmentation_layer: The image data augmentation layer
        cnn_units: The kernal sizes to use for the CNN layers
        cnn_filters: The filter sizes to use in the CNN layers
        cnn_strides: The strides to use in the CNN layers
        dropout: The dropout amount to use, Between 0-1
        dense_units: A list that defines the number of dense layers based on the number units provided for that index.
            The value at that index is the number of units for that layer.
        activation: A string for the activation function to use in all the dense layers.
        optimizer: The tensorflow class object that defines the optimizer to use.

    Returns:
        The tensorflow model.
    """

    tf.random.set_seed(15)
    keras.utils.set_random_seed(15)

    model_metrics = OrderedDict([
        ('accuracy', keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    ])
    
    # Model Architecture
    model = Sequential()

    if augmentation_layer is not None:
        model.add(augmentation_layer)

    
    # CNN Layers
    for i, units in enumerate(cnn_units):

        # First Convolution
        if (i == 0) and (augmentation_layer is None):
            model.add(layers.Conv2D(units, cnn_filters[i], strides=cnn_strides[i], input_shape=image_shape))
        else:
            model.add(layers.Conv2D(units, cnn_filters[i], strides=cnn_strides[i]))

        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))

        # Second Convolution
        model.add(layers.Conv2D(units, cnn_filters[i], strides=cnn_strides[i]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))

        # MaxPool
        model.add(layers.MaxPooling2D((2, 2)))

        if dropout:
            model.add(layers.Dropout(dropout))

            
    # Dense Layers
    model.add(layers.Flatten())
    for i, units in enumerate(dense_units):

        # Don't add activation to the final layer
        if i == (len(dense_units)-1):
            model.add(layers.Dense(units))
        else:
            model.add(layers.Dense(units, activation=activation))
    

    # Compile Model
    model.compile(optimizer=optimizer(learning_rate=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=list(model_metrics.values()))

    return model


def create_fit_and_save_model(model_name, train_df, val_df, test_df, params, fitting_batch_num):
    """
    Creates the tensormodel given the parameters.

    Args:
        model_name: The string name for the model.
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.
        epochs: The number of epochs for model training.
        params: A tuple with all model building parameters.
        fitting_batch_num: The batch number for this model fitting run.

    Returns:
        The validation metrics dictionary
    """

    batch_size, epochs, augmentation_params, cnn_params, dropout, dense_units, activation, optimizer, earlystop_patience, reducel_patience = params
    cnn_units, cnn_filters, cnn_strides = cnn_params
    
    if augmentation_params is not None:
        flip, rotate_factor, zoom_factor, random_flip_str = augmentation_params


    train_ds_str, val_ds_str, test_ds_str, train_ds, val_ds, test_ds = create_tensorflow_datasets(train_df, val_df, test_df, batch_size)

    image_batch, label_batch = next(iter(train_ds))
    image_shape = image_batch[0].shape

    if augmentation_params is not None:
        augmentation_layer = get_augmentation_layer(flip, rotate_factor, zoom_factor, random_flip_str, input_shape=image_shape)
    else:
        augmentation_layer = None

    model = build_model(
        image_shape=image_shape,
        augmentation_layer=augmentation_layer,
        cnn_units=cnn_units,
        cnn_filters=cnn_filters,
        cnn_strides=cnn_strides,
        dropout=dropout,
        dense_units=dense_units,
        activation=activation,
        optimizer=optimizer
    )

    model_output_dir = f'./model_checkpoints_{fitting_batch_num}'
    checkpoint_path = f"{model_output_dir}/{model_name}/cp.ckpt"
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_best_only=True, monitor='val_accuracy', mode='max')

    # Define parameters for early stopping and learning rate reduction
    earlystopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=earlystop_patience, verbose=0, restore_best_weights=False)
    reducel = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=reducel_patience, verbose=0, factor=0.5)

    start_time = time()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=[cp_callback, reducel, earlystopper],
                        verbose=0
                       )
    total_time = time() - start_time
    history_dict = save_history(history, model_output_dir, model_name)

    # Reload best weights
    model.load_weights(checkpoint_path)

    # Get validation metrics
    val_metrics = model.evaluate(val_ds, verbose=0)
    metrics = map_to_metrics(model.metrics_names, val_metrics)
    metrics['total_time'] = total_time
    metrics['ran_epochs'] = len(history.history['val_accuracy'])
    metrics['best_epoch'] = int(np.argmax(history_dict['val_accuracy']) + 1)

    json.dump(metrics, open(f"{model_output_dir}/val_metrics/{model_name}_val_metrics.json", 'w'), indent=4)

    return metrics


def send_ifttt_notification(value_send):
    """
    Sends and IFTTT notification

    Args:
        value_send: The string value to send in the notification.

    Returns:
        None. Notication will be sent if possible.
    """
    with open('ifttt_webhook.json') as json_file:
        json_data = json.load(json_file)

    ifttt_url = json_data['url']
    params = {"value1":value_send}
    r = requests.post(ifttt_url, params=params)

    if r.status_code == 200:
        print('Notification sent')
    else:
        print('Notification failure')


def save_history(history_obj, model_output_dir, MODEL_NAME):
    """
    Saves model history data as a JSON.

    Args:
        history_obj: The history object from the model.
        model_output_dir: The model output directory
        MODEL_NAME: The name of the model
    
    Returns:
        A dictionary of the history data.
    """

    history_dict = copy.deepcopy(history_obj.history)

    for k, v in history_dict.items():
        history_dict[k] = list(np.array(history_dict[k]).astype(float))
    
    if 'precision' in history_dict.keys() and 'recall' in history_dict.keys():
        pre = np.array(history_dict['precision'])
        rec = np.array(history_dict['recall'])
        history_dict['f1_score'] = list(2*(pre*rec/(pre+rec)))

        pre = np.array(history_dict['val_precision'])
        rec = np.array(history_dict['val_recall'])
        history_dict['val_f1_score'] = list(2*(pre*rec/(pre+rec+np.finfo(float).eps)))
        
    json.dump(history_dict, open(f"{model_output_dir}/model_histories/{MODEL_NAME}_history.json", 'w'), indent=4)

    return history_dict


def map_to_metrics(metrics_names, metric_tuple):
    """
    Maps a returned metric to its name.

    Args:
        metrics_names: The name of the metrics used for model fitting.
        metric_tuple: The tuple of model metrics
    
    Returns:
        A dictionary that maps a metric name to its value from the model.
    """

    return {key:value for key, value in zip(metrics_names, metric_tuple)}


def plot_metric(history:dict, metric_name:str, model_num=None, width=6):
    """
    Plots a model metric.

    Args:
        history: The model history dictionary containing the metrics
        metric_name: The name of the metric to plot.
        model_num: The number associated to this model.
        width: The width for the plot.
    """
    
    label_map = {
        'loss':'Loss',
        'accuracy':'Accuracy',
        'auc':'AUC',
        'f1_score':'F1 Score'
    }
    
    plt.figure(figsize=(width,4))
    plt.plot(history[metric_name], label=metric_name)
    plt.plot(history[f'val_{metric_name}'], label=f'val_{metric_name}')
    if model_num:
        plt.title(f"Model {model_num} Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    else:
        plt.title(f"Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    plt.xlabel('Epoch')
    plt.ylabel(label_map[metric_name])
    plt.xticks(range(len(history[metric_name])))
    plt.legend()
    plt.show()


def plot_learning_rate(history:dict, model_num=None, width=6):
    """
    Plots the learning rate.

    Args:
        history: The model history dictionary containing the metrics
        model_num: The number associated to this model.
        width: The width for the plot.
    """

    plt.figure(figsize=(width,4))
    plt.plot(history['lr'])
    if model_num:
        plt.title(f'Model {model_num} Learning Rate vs Epoch')
    else:
        plt.title('Learning Rate vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.xticks(range(len(history['lr'])))
    plt.show()


def bold(text):
    """
    Bold the input text for printing.

    Args:
        text: The input text.

    Returns:
        The text with the bold ANSI escape code.
    """

    return f"\033[1m{text}\033[0m"


def update_top_values(top_val_acc, val_acc, model_idx, fitting_batch_num, max_vals=10):
    """
    Updates the top_val_acc dictionary to maintain the top 10 best performing models.
    If a model isn't in the top 10, its checkpoint is deleted. (10 is the default number)

    Args:
        top_val_acc: A dictionary with the model index as the key and the validation
            accuracy for the value.
        val_acc: The validation accuracy for this model.
        model_idx: The current model index.
        fitting_batch_num:
        max_vals: The maximum number of checkpoints to save.
    
    Returns:
        None. Modifies the top_val_acc dictionary in-place.
    """
    
    removed_model = None
    if len(top_val_acc) < max_vals: # Keeping only the 10 best models.
        top_val_acc[model_idx] = val_acc
    elif val_acc > min(top_val_acc.values()):
        for k, v in top_val_acc.items():
            if v == min(top_val_acc.values()):
                del top_val_acc[k]
                removed_model = k
                break
        top_val_acc[model_idx] = val_acc
    else:
        removed_model = model_idx
    
    if removed_model is not None:
        shutil.rmtree(f'./model_checkpoints_{fitting_batch_num}/model{removed_model}', ignore_errors=True)


def load_results(model_checkpoint_dir:str):
    """
    Loads in the model results.

    Args:
        model_checkpoint_dir: The directory for the model checkpoint files.
    
    Returns:
        progress: A dataframe that tracked the model training progress. This dataframe shows the
            parameters used for the model.
        model_data: A dictionary that provides the model id, checkpoint location, and model history.
        all_val_metrics: A dataframe that contains the test metrics for all models loaded.
        full_results: A dataframe that combines the all_val_metrics and progress dataframes.
    """
    
    # During the model training, a model_building_progress file was created to track the training of all the models.
    progress = pd.read_csv(f'{model_checkpoint_dir}/model_building_progress.csv', keep_default_na=False)
    progress['model'] = progress['model'].apply(lambda x: f'model{x}')
    progress.set_index('model', inplace=True)
    if 'augmentation_params' in progress.columns:
        progress['augmentation_params'] = progress['augmentation_params'].apply(lambda x: x if x else 'None')

    # Reading in the model history and test results
    model_data = dict()
    all_val_metrics = []
    history_files = os.listdir(f'{model_checkpoint_dir}/model_histories')

    # Loading data for each trained model
    for file in history_files:
        model_name = file.split('_')[0]
        if re.match('model\d+', model_name):
            model_data[model_name] = {
                'id':int(re.findall('\d+', model_name)[0]),
                'checkpoint':f'{model_checkpoint_dir}/{model_name}/cp.ckpt',
                'model_history':pd.read_json(f'{model_checkpoint_dir}/model_histories/{model_name}_history.json'),
            }

            val_metrics = pd.DataFrame(json.load(open(f'{model_checkpoint_dir}/val_metrics/{model_name}_val_metrics.json')), index=[model_name])
            model_data[model_name]['val_metrics'] = val_metrics
            all_val_metrics.append(val_metrics)
    
    all_val_metrics = pd.concat(all_val_metrics)
    
    full_results = all_val_metrics.join(progress.astype(str))

    return progress, model_data, all_val_metrics, full_results


def calculate_performance(model, dataset):
    """
    Calculates the performance on the given dataset by creating a confusion matrix.

    Args:
        model: The tensorflow model
        dataset: The tensorflow dataset

    Returns:
        A tuple with (true labels, predicted labels,
            predicted probabilities, confusion matrix)
    """

    probability_model = tf.keras.Sequential([model, layers.Softmax()])
    pred_proba = probability_model.predict(dataset, verbose=0)
    y_pred = np.argmax(pred_proba, axis=1)

    labels = []
    for batch_images, batch_labels in dataset:
        labels.extend(batch_labels)
    
    y_true = np.array([l.numpy() for l in labels])

    # Get confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    matrix = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels = CLASS_NAMES)
    
    return y_true, y_pred, pred_proba, matrix