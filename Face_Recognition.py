import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam # Corrected: Adam is from keras.optimizers
import os
import traceback # For detailed error reporting

# --- GUI Specific Imports ---
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# --- Configuration ---
# IMPORTANT: Replace with your ACTUAL path to the dataset
TRAIN_DIR = r"project\Original Images\Original Images"
# Example alternative for non-Windows or if you prefer forward slashes:
# TRAIN_DIR = "path/to/your/project/Original Images/Original Images"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25 # Start with 25, adjust as needed based on training plots
VALIDATION_SPLIT = 0.2

# --- Global variables for GUI and model state ---
model = None
class_labels_ordered_by_index = []
num_classes = 0

# --- 1. Data Preparation ---
print("--- Preparing Data Generators ---")
try:
    if not os.path.exists(TRAIN_DIR) or not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: TRAIN_DIR '{TRAIN_DIR}' does not exist or is not a directory.")
        print("Please ensure the path is correct and the directory contains class subdirectories with images.")
        exit()

    # Training generator with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    # Validation generator (only rescaling)
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    train_ds = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_ds = validation_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    if not train_ds.classes.size: # Check if any classes were found
        print(f"ERROR: No classes found in {TRAIN_DIR}. Please check the directory structure.")
        print("Each class should be a subdirectory under TRAIN_DIR, containing images for that class.")
        exit()

    classes_from_generator = list(train_ds.class_indices.keys())
    num_classes = len(classes_from_generator) # Assign to global num_classes

    print(f"Found {train_ds.samples} images for training belonging to {num_classes} classes.")
    print(f"Found {validation_ds.samples} images for validation belonging to {num_classes} classes.")
    print(f"Class labels (from generator): {classes_from_generator}")
    print(f"Class indices: {train_ds.class_indices}")

    # --- Create an ordered list of class names based on Keras's indices ---
    class_labels_ordered_by_index = [None] * num_classes # Assign to global
    for class_name_iter, index_iter in train_ds.class_indices.items():
        if 0 <= index_iter < num_classes:
            class_labels_ordered_by_index[index_iter] = class_name_iter
        else:
            print(f"Warning: Index {index_iter} for class {class_name_iter} is out of bounds for num_classes {num_classes}")

    if None in class_labels_ordered_by_index:
        print(f"Warning: Not all class labels were correctly mapped. Ordered list: {class_labels_ordered_by_index}")
        if len(classes_from_generator) == num_classes:
             class_labels_ordered_by_index = classes_from_generator
        else:
            print("ERROR: Critical issue in mapping class labels. Exiting.")
            exit()
    print(f"Ordered class labels for prediction GUI: {class_labels_ordered_by_index}")

except Exception as e:
    print(f"Error during data preparation: {e}")
    traceback.print_exc()
    # Allow script to continue to GUI launch, but model will be None
    # exit() # Or uncomment to stop if data prep is critical

# --- 2. Define CNN Model ---
if num_classes > 0: # Only define model if classes were found
    print("\n--- Defining CNN Model ---")
    model = Sequential([ # Assign to global model
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # --- 3. Compile the Model ---
    print("\n--- Compiling Model ---")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    model.summary()

    # --- 4. Train the Model ---
    print("\n--- Starting Model Training ---")
    history = None # Initialize history

    steps_per_epoch = train_ds.samples // BATCH_SIZE
    if train_ds.samples % BATCH_SIZE != 0:
        steps_per_epoch += 1
    if train_ds.samples > 0 and steps_per_epoch == 0:
        steps_per_epoch = 1

    validation_steps = validation_ds.samples // BATCH_SIZE
    if validation_ds.samples % BATCH_SIZE != 0:
        validation_steps += 1
    if validation_ds.samples > 0 and validation_steps == 0:
        validation_steps = 1

    if train_ds.samples == 0:
        print("WARNING: No training images found. Skipping training.")
        # model will remain None or uncompiled if it was defined
    else:
        try:
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_ds if validation_ds.samples > 0 else None,
                validation_steps=validation_steps if validation_ds.samples > 0 else None
            )
            print("--- Model Training Finished ---")
        except Exception as e:
            print(f"Error during model training: {e}")
            traceback.print_exc()
            # model might be partially trained or still in initial state

    # --- 5. Evaluate and Plot Results ---
    if history and hasattr(history, 'history') and history.history:
        print("\n--- Plotting Training History ---")
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        print("IMPORTANT: Close the plot window to continue to the GUI.")
        plt.show() # This is a blocking call

        if validation_ds.samples > 0 and 'val_accuracy' in history.history:
            print("\n--- Evaluating on Validation Set ---")
            val_loss, val_accuracy = model.evaluate(validation_ds, steps=validation_steps)
            print(f"Final Validation Loss: {val_loss:.4f}")
            print(f"Final Validation Accuracy: {val_accuracy:.4f}")
else:
    print("Skipping model definition, compilation, and training due to no classes found or data preparation error.")


# --- 6. GUI for Image Prediction ---

# Global variables for GUI elements that need updating
image_label_tk = None
actual_class_label_tk = None
predicted_class_label_tk = None
confidence_label_tk = None
probabilities_text_tk = None

def predict_from_gui(image_path):
    """Loads, preprocesses, and predicts an image, then updates GUI."""
    global image_label_tk, actual_class_label_tk, predicted_class_label_tk, confidence_label_tk, probabilities_text_tk
    global class_labels_ordered_by_index, model # Use the globally defined variables

    if not image_path:
        return

    if model is None:
        actual_class_label_tk.config(text="Error: Model is not available for prediction.")
        predicted_class_label_tk.config(text="")
        confidence_label_tk.config(text="")
        return

    try:
        # Display the selected image
        img_pil = Image.open(image_path)
        img_pil_display = img_pil.copy()
        img_pil_display.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil_display)
        image_label_tk.config(image=img_tk)
        image_label_tk.image = img_tk

        # Preprocess for model
        img_for_model = load_img(image_path, target_size=IMAGE_SIZE)
        x = img_to_array(img_for_model)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Predict
        predictions_array = model.predict(x)[0]

        # Get actual class
        parent_dir = os.path.dirname(image_path)
        actual_class = os.path.basename(parent_dir)

        # Get predicted class
        predicted_index = np.argmax(predictions_array)

        if 0 <= predicted_index < len(class_labels_ordered_by_index) and class_labels_ordered_by_index[predicted_index] is not None:
            predicted_class_name = class_labels_ordered_by_index[predicted_index]
        else:
            predicted_class_name = f"Unknown Index ({predicted_index})"
            print(f"Warning: Predicted index {predicted_index} is out of bounds or unmapped for class_labels_ordered_by_index.")

        confidence = predictions_array[predicted_index]

        actual_class_label_tk.config(text=f"Actual Class: {actual_class}")
        predicted_class_label_tk.config(text=f"Predicted Class: {predicted_class_name}")
        confidence_label_tk.config(text=f"Confidence: {confidence:.4f}")

        probs_text = "Prediction Probabilities:\n"
        for i, prob in enumerate(predictions_array):
            class_name_display = class_labels_ordered_by_index[i] if 0 <= i < len(class_labels_ordered_by_index) and class_labels_ordered_by_index[i] is not None else f"Class {i}"
            probs_text += f"- {class_name_display}: {prob:.4f}\n"

        probabilities_text_tk.config(state=tk.NORMAL)
        probabilities_text_tk.delete(1.0, tk.END)
        probabilities_text_tk.insert(tk.END, probs_text)
        probabilities_text_tk.config(state=tk.DISABLED)

    except Exception as e:
        error_message = f"Error: {str(e)[:200]}"
        actual_class_label_tk.config(text=error_message)
        predicted_class_label_tk.config(text="")
        confidence_label_tk.config(text="")
        if probabilities_text_tk:
            probabilities_text_tk.config(state=tk.NORMAL)
            probabilities_text_tk.delete(1.0, tk.END)
            probabilities_text_tk.insert(tk.END, f"Error processing image: {str(e)[:200]}")
            probabilities_text_tk.config(state=tk.DISABLED)
        print(f"Error during GUI prediction for {image_path}:")
        traceback.print_exc()


def select_image_and_predict():
    """Opens a file dialog to select an image and then predicts."""
    initial_dir_suggestion = TRAIN_DIR if os.path.exists(TRAIN_DIR) and os.path.isdir(TRAIN_DIR) else os.getcwd()
    if os.path.isdir(initial_dir_suggestion): # If TRAIN_DIR is a dir, go one level up for better browsing
        parent_of_train_dir = os.path.dirname(initial_dir_suggestion)
        if os.path.isdir(parent_of_train_dir):
            initial_dir_suggestion = parent_of_train_dir

    file_path = filedialog.askopenfilename(
        initialdir=initial_dir_suggestion,
        title="Select an Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
    )
    if file_path:
        predict_from_gui(file_path)

def create_prediction_gui():
    global image_label_tk, actual_class_label_tk, predicted_class_label_tk, confidence_label_tk, probabilities_text_tk
    global num_classes # Access the global num_classes

    root = tk.Tk()
    root.title("Image Classifier")
    min_width = 750
    min_height = 550
    root.minsize(min_width, min_height)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    image_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
    image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

    image_label_tk = tk.Label(image_frame, text="Selected image will appear here", font=("Arial", 10), anchor="center")
    image_label_tk.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)

    controls_results_frame = tk.Frame(main_frame)
    controls_results_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

    controls_frame = tk.Frame(controls_results_frame, pady=10)
    controls_frame.pack(side=tk.TOP, fill=tk.X)

    select_button = tk.Button(controls_frame, text="Select Image & Predict", command=select_image_and_predict, font=("Arial", 12), padx=10, pady=5)
    select_button.pack()

    results_frame = tk.Frame(controls_results_frame, padx=10)
    results_frame.pack(side=tk.TOP, fill=tk.X, expand=False, anchor=tk.N)

    tk.Label(results_frame, text="--- Results ---", font=("Arial", 14, "bold")).pack(pady=(10,10))

    actual_class_label_tk = tk.Label(results_frame, text="Actual Class: ", font=("Arial", 12), anchor="w", justify=tk.LEFT)
    actual_class_label_tk.pack(pady=2, fill=tk.X)

    predicted_class_label_tk = tk.Label(results_frame, text="Predicted Class: ", font=("Arial", 12), anchor="w", justify=tk.LEFT)
    predicted_class_label_tk.pack(pady=2, fill=tk.X)

    confidence_label_tk = tk.Label(results_frame, text="Confidence: ", font=("Arial", 12), anchor="w", justify=tk.LEFT)
    confidence_label_tk.pack(pady=2, fill=tk.X)

    tk.Label(results_frame, text="--------------------", font=("Arial", 10)).pack(pady=(5,0))

    # Determine height of probabilities text box
    # Use num_classes if available and > 0, otherwise a default
    text_height = max(5, num_classes if num_classes > 0 else 5)

    probabilities_text_tk = tk.Text(results_frame, height=text_height, width=35, font=("Arial", 10), wrap=tk.WORD)
    probabilities_text_tk.pack(pady=5, fill=tk.X, expand=False)
    probabilities_text_tk.insert(tk.END, "Prediction Probabilities will appear here.")
    probabilities_text_tk.config(state=tk.DISABLED)

    root.mainloop()

# --- Run the GUI (after training or if model is loaded) ---
if __name__ == "__main__": # <<<< ****** CORRECTED THIS LINE ******
    # Training and plotting will happen first (if data is available).
    # IMPORTANT: You need to CLOSE the matplotlib plot window for the GUI to launch.

    # Check if model and class labels are somewhat ready for the GUI
    if model is None:
        print("WARNING: Model was not trained or loaded. GUI prediction will not function.")
        # You could attempt to load a pre-saved model here if desired:
        # try:
        #     print("Attempting to load a pre-saved model...")
        #     model = tf.keras.models.load_model("my_image_classifier.h5") # Provide actual path
        #     # If loading a model, you also need to ensure class_labels_ordered_by_index and num_classes are set.
        #     # This might involve loading them from a file or knowing the model's output structure.
        #     # Example:
        #     # num_classes = model.layers[-1].output_shape[-1]
        #     # class_labels_ordered_by_index = ["class0", "class1", ...] # Manually set or load
        #     print("Loaded a pre-saved model.")
        # except Exception as e_load:
        #     print(f"Could not load a pre-saved model: {e_load}. GUI will launch with limited functionality.")
        pass # Allow GUI to launch even if model is None

    if not class_labels_ordered_by_index or None in class_labels_ordered_by_index:
        print("WARNING: Class labels not properly initialized. GUI might not display class names correctly.")
        # Attempt a basic fallback if num_classes is known
        if num_classes > 0:
            print(f"Attempting fallback for class labels using num_classes: {num_classes}")
            class_labels_ordered_by_index = [f"Class {i}" for i in range(num_classes)]
        elif model and hasattr(model, 'layers') and model.layers: # Try to get from model if it exists
             try:
                output_layer_shape = model.layers[-1].output_shape
                if isinstance(output_layer_shape, tuple) and len(output_layer_shape) > 1 and isinstance(output_layer_shape[-1], int):
                    num_classes_from_model = model.layers[-1].output_shape[-1]
                    if num_classes_from_model > 0:
                        print(f"Attempting fallback for class labels using num_classes from model output: {num_classes_from_model}")
                        class_labels_ordered_by_index = [f"Class {i}" for i in range(num_classes_from_model)]
                        num_classes = num_classes_from_model
             except: pass # Ignore errors in this fallback
        if not class_labels_ordered_by_index or None in class_labels_ordered_by_index: # Final fallback
            print("Using generic class labels as a last resort.")
            class_labels_ordered_by_index = ["Unknown Class"]
            if num_classes == 0: num_classes = 1 # Ensure probabilities text box has a sensible default height


    print("\nLaunching Image Prediction GUI...")
    create_prediction_gui()

print("\n--- Script Finished ---")
