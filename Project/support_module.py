import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import keras

TARGET_SIZE = (257, 344)

# Reading and resizing images. Preprocessing for ResNet and for Custom CNN model
def preprocess_image(image_path, measurements, source):
    # Read the image
    image = tf.io.read_file(source+image_path)
    image = tf.image.decode_png(image, channels=3)

    # Resize the image to match ResNet50 input
    image = tf.image.resize(image, [257, 344])

    # Resize the image to a square by cropping the central part, then to 224x224 for Custom model
    # image = tf.image.resize_with_crop_or_pad(image, 344, 344)  # Crop to square
    # image = tf.image.resize(image, [224, 224])  # Resize to the desired input size

    # Normalize the image
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, measurements

# def load_and_preprocess_image(image_path, target_size):
#     """
#     Load and preprocess the image.
#     """
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = tf.keras.applications.resnet50.preprocess_input(img)
#     return img


def load_and_preprocess_image(image_path, target_size, use_resnet_preprocessing):
    """
    Load and preprocess the image. Applies ResNet50 preprocessing if specified.

    :param image_path: Path to the image file.
    :param target_size: The target size to resize the image.
    :param use_resnet_preprocessing: Boolean, whether to apply ResNet50 preprocessing.
    :return: Preprocessed image array suitable for model prediction.
    """
    # Load the image and convert it to an array
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Expand the image array to include a batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Apply preprocessing specific to the model type
    if use_resnet_preprocessing:
        img = tf.keras.applications.resnet50.preprocess_input(img)
    else:
        # For non-ResNet models, adjust according to the model's expected preprocessing.
        # This example normalizes pixel values to the range [0, 1] as a generic approach.
        img = img / 255.0
    
    return img


def preprocess_image_custom(image_path, measurements, source):
    # Read the image
    image = tf.io.read_file(source + image_path)
    image = tf.image.decode_png(image, channels=3)

    # Resize the image to a square by cropping the central part, then to 224x224
    image = tf.image.resize_with_crop_or_pad(image, 344, 344)  # Crop to square
    image = tf.image.resize(image, [224, 224])  # Resize to the desired input size

    # Normalize the image (update this if you have a specific way of normalizing for your custom model)
    image = tf.cast(image, tf.float32) / 255.0

    return image, measurements

def create_test_dataset(dataframe, source, batch_size):
    image_paths = dataframe["filename"].values
    measurements = dataframe[["chest_circ", "waist_circ", "pelvis_circ", "neck_circ", "bicep_circ", "thigh_circ", "knee_circ", "arm_length", "leg_length", "calf_length", "head_circ", "wrist_circ", "arm_span", "shoulders_width", "torso_length", "inner_leg"]].values

    # Creating the tf Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, measurements))

    # Map the dataset to the preprocessing function
    test_dataset = dataset.map(lambda x, y: preprocess_image_custom(x, y, source), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_dataset

def create_testset(dataframe, source, batch_size, train_size, shuffle_buffer_size):
    image_paths = dataframe["filename"].values
    measurements = dataframe[["chest_circ", "waist_circ", "pelvis_circ", "neck_circ", "bicep_circ", "thigh_circ", "knee_circ",	"arm_length", "leg_length", "calf_length", "head_circ", "wrist_circ",	"arm_span", "shoulders_width", "torso_length", "inner_leg"]].values

    #Creating the tf Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, measurements))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42, reshuffle_each_iteration=True)

    num_train_samples = int(len(dataset) * train_size)
    test_dataset = dataset.take(num_train_samples).map(lambda x, y: preprocess_image_custom(x, y, source),num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset

def create_test_dataset_resnet(dataframe, source, batch_size):
    image_paths = dataframe["filename"].values
    measurements = dataframe[["chest_circ", "waist_circ", "pelvis_circ", "neck_circ", "bicep_circ", "thigh_circ", "knee_circ", "arm_length", "leg_length", "calf_length", "head_circ", "wrist_circ", "arm_span", "shoulders_width", "torso_length", "inner_leg"]].values

    # Creating the tf Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, measurements))

    # Map the dataset to the preprocessing function
    test_dataset = dataset.map(lambda x, y: preprocess_image(x, y, source), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_dataset

def create_heatmap(model, img, layer_name):
    """
    Generate heatmap from the model and input image.
    """
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Pooling and heatmap creation
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # Relu and resize heatmap
    cam = np.maximum(cam, 0)
    heatmap = cv2.resize(cam, TARGET_SIZE)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def superimpose_heatmap(heatmap, original_img):
    """
    Superimpose the heatmap on the original image.
    """
    heatmap = cv2.resize(heatmap, (original_img.shape[2], original_img.shape[1]))
    original_img = np.squeeze(original_img)
    superimposed_img = heatmap * 0.4 + original_img
    return np.uint8(superimposed_img)


def predict_measurements_and_evaluate(dataset, model, source_folder, target_size=(224, 224), resnet=False):
    """
    Predicts measurements from images in the given DataFrame and evaluates the accuracy.
    
    :param dataset: DataFrame containing 'filename' and measurement columns.
    :param model: Pre-trained Keras model for prediction.
    :param source_folder: Path to the folder containing images.
    :param target_size: Tuple for resizing images, default is (224, 224).
    :param resnet: Boolean indicating if ResNet preprocessing should be applied.
    :return: DataFrame with predicted measurements and prints MAE, MSE for each column.
    """
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()

    predicted_measurements = []

    for _, row in dataset.iterrows():
        try:
            img_path = source_folder + row['filename']
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, target_size)
            img_array = tf.keras.applications.resnet50.preprocess_input(img) if resnet else img / 255.0
            img_array = tf.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array, verbose=0)
            predicted_measurements.append(prediction[0])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            predicted_measurements.append([None] * len(dataset.columns[1:-1]))  # Adjust None values based on your model's output shape

    predicted_df = pd.DataFrame(predicted_measurements, columns=dataset.columns[1:-1])

    # Calculate and print accuracy metrics
    all_mae, all_mse = [], []
    for col in predicted_df.columns:
        # Filter out None values if any errors occurred during processing
        valid_idxs = ~predicted_df[col].isnull()
        true_values = dataset[col][valid_idxs].values
        pred_values = predicted_df[col][valid_idxs].values
        mae_value = mae(true_values, pred_values).numpy()
        mse_value = mse(true_values, pred_values).numpy()
        all_mae.append(mae_value)
        all_mse.append(mse_value)
        print(f"MAE for {col}: {mae_value}, MSE for {col}: {mse_value}")

    mean_MAE = sum(all_mae) / len(all_mae)
    mean_MSE = sum(all_mse) / len(all_mse)
    print(f"Mean MAE : {mean_MAE}, Mean MSE : {mean_MSE}")
    return predicted_df


def visualize_feature_maps(model, layer_name, input_image, num_maps=256, rows=32, cols=8):
    # Create a model that outputs the specified layer's output
    layer_output = model.get_layer(layer_name).output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)

    # Generate the feature maps
    feature_maps = intermediate_model.predict(input_image)

    # Define the number of feature maps to display
    num_feature_maps = feature_maps.shape[-1]
    displayed_maps = min(num_maps, num_feature_maps)

    # Plot the feature maps in larger individual subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # Increase the figsize for larger images
    axes = axes.flatten()
    for i in range(displayed_maps):
        ax = axes[i]
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')
    for i in range(displayed_maps, rows * cols):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Example usage:
# visualize_feature_maps(your_model, 'name_of_conv_layer', your_input_image, num_maps=16, rows=4, cols=4)
@keras.saving.register_keras_serializable()
def custom_loss_function(y_true, y_pred):
    threshold = 0.1
    error = tf.abs(y_true - y_pred)
    higher_penalty = 2.0
    loss = tf.where(error > threshold, higher_penalty * error, error)
    return tf.reduce_mean(loss, axis=-1)

def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        img = load_and_preprocess_image(IMAGE_PATH, TARGET_SIZE)
        heatmap = create_heatmap(model, img)
        superimposed_img = superimpose_heatmap(heatmap, img)
        cv2.imwrite(SAVE_PATH, superimposed_img)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
