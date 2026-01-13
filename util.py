import os
import glob
import random
import re
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import sys


SEED = 42
def set_seeds():
    # Seed for reproducibility
    # Set the seed for the Python random module
    random.seed(SEED)
    # Set the seed for NumPy
    np.random.seed(SEED)
    # Ensure reproducibility with certain environment variables
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # sets the TensorFlow seed
    tf.keras.utils.set_random_seed(SEED)
    # determinism in TensorFlow OPS for reproducibility
    tf.config.experimental.enable_op_determinism()

# Redirect sys.stdout to suppress progress bars
class NoOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_2d_array_from_slice_png(png_path):
    return np.array(Image.open(png_path).convert('L'))

def load_slices_from_dir(scan_dir):
    slice_files = glob.glob(f'{scan_dir}/*.png')
    return [load_2d_array_from_slice_png(file) for file in slice_files]

def load_slices_from_dir_and_label_lists(scan_folder_list, label_list):
    all_slices = []
    all_labels = []
    for scan_dir, label in zip(scan_folder_list, label_list):
        slices = load_slices_from_dir(scan_dir)
        all_slices.extend(slices)
        all_labels.extend([label] * len(slices))
    return np.array(all_slices), np.array(all_labels)

# resize a 2D image to the target size (using TF)
def resize_image(image, target_size=(192, 192)):
    """
    Resizes a 2D image to the target_size.
    Converts the image to a tensor, resizes it, and converts it back to a numpy array.
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    # If image is grayscale without channel dimension, add one temporarily
    if len(image_tensor.shape) == 2:
        image_tensor = tf.expand_dims(image_tensor, -1)
    resized = tf.image.resize(image_tensor, target_size)
    # Squeeze the channel dimension if it was added
    if resized.shape[-1] == 1:
        resized = tf.squeeze(resized, axis=-1)
    return resized.numpy()

def min_max_normalize(tensor):
    """Min-max normalizes a tensor."""
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    return (tensor - min_val) / (max_val - min_val)  

def min_max_normalize_np(image):
    return (image - image.min()) / (image.max() - image.min())

def preprocess_slice(image):
    image = image.astype(np.float32)
    return min_max_normalize(image)

def preprocess_slice_np(image):
    image = image.astype(np.float32)
    return min_max_normalize_np(image)

# --- load slices from one scan and record metadata ---
def load_slices_from_scan(scan_dir, resize=False):
    """
    For a given scan directory, load all .png slices and extract metadata.
    Assumes that the directory structure is like:
      .../MRI/<prefix>__<dataset>/<scan_id>
    and that each slice image is named like "slice_066.png" (slice index).
    """
    # Get all png files in the scan directory
    slice_files = glob.glob(os.path.join(scan_dir, '*.png'))
    
    # Extract dataset and scan_id from the path.
    # For example: ./MRI/_MS__ISBI_3T_test/01_01
    # we split the path and take the last two parts.
    parts = scan_dir.split(os.sep)
    # second-to-last folder (e.g., "_MS__ISBI_3T_test")
    dataset_dir = parts[-2]
    # last folder (e.g., "01_01")
    scan_id = parts[-1]
    
    # Clean the dataset name by taking the part after "__"
    dataset = dataset_dir.split("__")[-1]
    
    # For some datasets (e.g., healthy scans) the scan_id might have a prefix like "Guys-"
    # Remove any non-numeric prefix if desired.
    if "-" in scan_id:
        scan_id = scan_id.split("-")[-1]
    
    slices = []
    metadata = []
    for slice_file in slice_files:
        # Get the slice index from the filename (e.g., 66 from "slice_066.png")
        slice_idx = int(os.path.splitext(os.path.basename(slice_file))[0].split('_')[-1])
        # Load and preprocess the image
        image = load_2d_array_from_slice_png(slice_file)
        image = preprocess_slice(image)
        if resize:
            image = resize_image(image, target_size=(192, 192))
        slices.append(image)
        metadata.append({
            'dataset': dataset,
            'scan_id': scan_id,
            'slice_idx': slice_idx
        })
    return slices, metadata

def load_slices_from_scan_np(scan_dir, resize=False):
    """
    Loads slices and metadata from a scan directory.
    """
    slice_files = glob.glob(os.path.join(scan_dir, '*.png'))
    parts = scan_dir.split(os.sep)
    dataset_dir = parts[-2]
    scan_id = parts[-1]
    dataset = dataset_dir.split("__")[-1]
    if "-" in scan_id:
        scan_id = scan_id.split("-")[-1]
    
    slices = []
    metadata = []
    for slice_file in slice_files:
        slice_idx = int(os.path.splitext(os.path.basename(slice_file))[0].split('_')[-1])
        # Load slice image
        image = load_2d_array_from_slice_png(slice_file)
        # Use NumPy-based preprocessing
        image = preprocess_slice_np(image)
        if resize:
            image = resize_image(image, target_size=(192, 192))
        slices.append(image)
        metadata.append({
            'dataset': dataset,
            'scan_id': scan_id,
            'slice_idx': slice_idx
        })
    return slices, metadata

# def predict_scans(scan_dirs, class_labels, model):
#     """
#     For each scan directory, load slices, resize slices to (192, 192), perform batch predictions 
#     with the given TF model, and return a DataFrame with the following columns:
#       - dataset, scan_id, slice_idx, class, predicted_class, pred_prob_0, pred_prob_1, is_correct.
    
#     Parameters:
#       scan_dirs (list of str): Directories containing scan slices.
#       class_labels (list of int): The true class label for each scan (0 or 1).
#       model: A TensorFlow model for making predictions.
#     """
#     results = []
    
#     for scan_dir, class_val in zip(scan_dirs, class_labels):
#         slices, meta = load_slices_from_scan_np(scan_dir)
#         transformed_slices = []
        
#         # resize each slice
#         for img in slices:
#             img = resize_image(img, target_size=(192, 192))
#             transformed_slices.append(img)
        
#         # Convert list of slices to a batch array.
#         batch = np.stack(transformed_slices, axis=0)
        
#         # Batch predict using the TensorFlow model
#         pred_probs = model.predict(batch)  # assuming 'metadata' exists per sample
#         pred_classes = np.argmax(pred_probs, axis=1)
#         is_correct = (pred_classes == class_val)
        
#         # Merge metadata with predictions
#         for m, probs, p_class, correct in zip(meta, pred_probs, pred_classes, is_correct):
#             m['class'] = class_val
#             m['predicted_class'] = int(p_class)
#             m['is_correct'] = bool(correct)
#             m['pred_prob_0'] = float(probs[0])
#             m['pred_prob_1'] = float(probs[1])
#             m['actual_class_pred_prob'] = m['pred_prob_0'] * (1 - m['class']) + m['pred_prob_1'] * m['class']
#             results.append(m)
    
#     df = pd.DataFrame(results)
#     return df
def predict_scans(scan_dirs, class_labels, model, include_logits=False, include_embeddings=False):
    """
    Predict scan slices and return a DataFrame with optional logits and embeddings.

    Parameters:
      scan_dirs (list of str): Directories containing scan slices.
      class_labels (list of int): The true class labels for each scan (0 or 1).
      model: A TensorFlow model for making predictions.
      include_logits (bool): Whether to include logits in output.
      include_embeddings (bool): Whether to include embeddings in output.

    Returns:
      pd.DataFrame: DataFrame with prediction results.
    """
    all_results = []

    # Build intermediate outputs explicitly
    intermediate_outputs = []

    if include_embeddings:
        embedding_output = tf.keras.layers.GlobalMaxPooling2D(name='embeddings')(model.get_layer('dropout_conv4').output)
        intermediate_outputs = [embedding_output]

    if include_logits:
        intermediate_outputs.append(model.get_layer('logits').output)

    intermediate_outputs.append(model.output)  # Softmax predictions always included

    intermediate_model = tf.keras.Model(inputs=model.input, outputs=intermediate_outputs)

    for scan_dir, actual_class in zip(scan_dirs, class_labels):
        slices, meta = load_slices_from_scan_np(scan_dir)
        slices_resized = np.array([resize_image(slice_img, (192, 192)) for slice_img in slices])

        predictions = intermediate_model.predict(slices_resized)

        idx = 0
        if include_embeddings:
            embeddings_pred = predictions[idx]
            idx += 1

        if include_logits:
            logits_pred = predictions[idx]
            idx += 1

        if include_embeddings or include_logits:
            pred_probs = predictions[idx]
        else:
            pred_probs = predictions

        for i, probs in enumerate(pred_probs):
            result = meta[i].copy()
            predicted_class = int(probs.argmax())

            result.update({
                'class': actual_class,
                'predicted_class': predicted_class,
                'is_correct': predicted_class == actual_class,
                'pred_prob_0': float(probs[0]),
                'pred_prob_1': float(probs[1]),
                'actual_class_pred_prob': float(probs[actual_class]),
            })

            if include_logits:
                result.update({
                    'logit_0': float(predictions[idx - (2 if include_embeddings else 1)][i][0]),
                    'logit_1': float(predictions[idx - 1][i][1]),
                })

            if include_embeddings:
                result['embedding'] = embeddings_pred[i].tolist()

            all_results.append(result)

    return pd.DataFrame(all_results)

def write_paths_to_file(file_path, paths):
    with open(file_path, "w") as f:
        f.writelines(f"{path}\n" for path in paths)

def read_paths_from_file(file_path):
    with open(file_path, "r") as f:
        paths_read = [line.strip() for line in f]
    return paths_read

def add_relative_slice_idx_col(df):
    df['relative_slice_idx'] = df.groupby('scan_id')['slice_idx'].transform(lambda x: x - x.min())

def extract_data_variant(model_str):
    patterns = ['baseline', 'blurred_SD[1-4]', 'contrast_(?:imadjust|histeq|adapthisteq)']
    match = re.search('|'.join(patterns), model_str)
    return match.group(0) if match else None

def parse_model_and_add_data_variant_col(df, model):
    df = df.copy()
    df['model'] = model
    df['data_variant'] = df['model'].apply(extract_data_variant)
    return df #.drop('model', axis=1)

def select_calibration_ids_with_class_check(ids_cal, st, NUM_SELECT, run, max_attempts=1000):
    """
    Try up to `max_attempts` times to sample calibration IDs that include both classes.
    Increments the seed by 1000 on each retry attempt.
    """
    for attempt in range(max_attempts):
        current_seed = run + attempt * 1000
        rng = np.random.default_rng(current_seed)
        cal_ids = rng.choice(ids_cal, NUM_SELECT, replace=False)

        # Check if both classes are present in the selected calibration set
        cal_slice = st.cal_df.query("scan_id in @cal_ids")
        classes_present = cal_slice['class'].unique()

        if set(classes_present) >= {0, 1}:
            return cal_ids, current_seed  # Successful sample with both classes

    # If no valid sample found after all attempts, raise an error
    raise ValueError(
        f"Unable to select calibration IDs with both classes present after {max_attempts} attempts for run {run}."
    )

