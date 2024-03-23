import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

CLASS_NAMES = ['grizzly', 'black', 'teddy']

def prep_data(data_dir:str):
    """
    Preps data files for training.

    Args:
        data_dir: The main directory the data is in.
    
    Returns:
        df: DataFrame of the training labels.
    """

    # Read in files and create a dataframe to handle file mapping
    if os.path.exists(f'{data_dir}/bears_dataset.csv'):
        df = pd.read_csv(f'{data_dir}/bears_dataset.csv')
    else:
        data = []
        for class_name in CLASS_NAMES:
            for file in os.listdir(f'{data_dir}/{class_name}'):
                data.append((class_name, file))

        df = pd.DataFrame.from_records(data, columns=['label', 'file'])
        df.sort_values(['label', 'file'], ignore_index=True, inplace=True)
        df.to_csv(f'{data_dir}/dataset.csv', index=False)
    
    df['path'] = f'{data_dir}/' + df['label'] + '/' + df['file']
    
    return df


def get_train_val_test_data(input_df:pd.DataFrame):
    """
    Creates the training, validation, and test data splits.

    Args:
        input_df: The DataFrame to split
    
    Returns:
        (train_df, val_df, test_df): The training, validation, and test datasets
    """

    train_df, test_val_df = train_test_split(input_df, test_size=0.3, stratify=input_df['label'], random_state=15)
    val_df, test_df = train_test_split(test_val_df, test_size=0.25, stratify=test_val_df['label'], random_state=15)

    print(f'Training set: 70%, Validation set: {0.3*0.75:.1%}, Test set: {0.3*0.25:.1%}')

    return train_df, val_df, test_df


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == CLASS_NAMES

    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # Load image and keep only 3 channels
    img = tf.image.decode_jpeg(img, channels=3)
    # Normalize pixel values
    img = tf.cast(img, tf.float16)
    img = (img / 255.0)

    return img


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    
    return img, label


def process_path_no_label(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img


def configure_for_performance(ds, batch_size, shuffle=True):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache()

    #Turn off this shuffle so that images and labels could be re-mapped together
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=15) 
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_tensorflow_datasets(train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame, batch_size:int):
    """
    Converts datasets into compatible tensorflow datasets.

    Args:
        train_df: The training dataset
        val_df: The validation dataset
        test_df: The test dataset
        batch_size: The batch size to use during training.

    Returns:
        train_ds_str, val_ds_str, test_ds_str: String versions 
            of the training, validation, and test datasets 
        train_ds, val_ds, test_ds: The final versions
            of the training, validation, and test datasets
    """

    train_ds_str = tf.data.Dataset.from_tensor_slices(train_df['path'].values)
    val_ds_str = tf.data.Dataset.from_tensor_slices(val_df['path'].values)
    test_ds_str = tf.data.Dataset.from_tensor_slices(test_df['path'].values)

    AUTOTUNE = tf.data.AUTOTUNE

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = configure_for_performance(train_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                         batch_size)
    val_ds = configure_for_performance(val_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                       batch_size)
    test_ds = configure_for_performance(test_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                        batch_size, shuffle=False)

    return train_ds_str, val_ds_str, test_ds_str, train_ds, val_ds, test_ds


def verify_pipeline(dataset, are_pixels_normalized=True):
    """
    This will verify that the tensorflow datasets are working.

    Args:
        dataset: The tensorflow dataset.
        are_pixels_normalized: If true, the images in the dataset have been
            normalized between 0-1.
    
    Returns:
        None. Plots example images.
    """

    image_batch, label_batch = next(iter(dataset))

    if are_pixels_normalized:
        factor = 255
    else:
        factor = 1

    plt.figure(figsize=(10, 10))
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((image_batch[i].numpy()*factor).astype("uint8"))
        label = label_batch[i]
        plt.title(CLASS_NAMES[label])
        plt.axis("off")