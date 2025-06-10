# Step 0: Install Dependencies (run this in your VS Code terminal if needed)
# pip install pandas scikit-learn joblib matplotlib seaborn PyMuPDF python-docx tensorflow
#
# For Tkinter (usually included, but if not):
# On Debian/Ubuntu: sudo apt-get install python3-tk
# On Fedora: sudo dnf install python3-tkinter

# Step 1: Imports
print("\nImporting libraries...")
import pandas as pd
import joblib
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # GlobalMaxPooling1D, Conv1D can be alternatives
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Imports for text extraction and file dialog
import fitz  # PyMuPDF
import docx  # python-docx
from tkinter import Tk, filedialog

# Optional: Suppress warnings and plotting setup
import matplotlib.pyplot as plt
# import seaborn as sns # Not actively used in this version, can be added if needed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO/WARNING messages
print("Libraries imported.")

# --- Configuration for Deep Learning Model ---
VOCAB_SIZE = 10000  # Max number of words to keep in the vocabulary
MAX_LENGTH = 250    # Max length of input sequences (increased slightly)
EMBEDDING_DIM = 100 # Dimension of word embeddings
LSTM_UNITS = 128    # Number of units in LSTM layer
# --- End Configuration ---


# Step 2: Define Text Extraction Function
print("\nDefining helper functions...")
def extract_text(file_path):
    """Extracts text from PDF, DOCX, or TXT files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found during extraction: {file_path}")

    if file_path.lower().endswith(".pdf"):
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
            return text
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {file_path}: {e}") from e

    elif file_path.lower().endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
             raise RuntimeError(f"Error processing DOCX {file_path}: {e}") from e

    elif file_path.lower().endswith(".txt"):
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise RuntimeError(f"Could not decode TXT file {file_path} with common encodings.")
        except Exception as e:
             raise RuntimeError(f"Error processing TXT {file_path}: {e}") from e
    else:
        _, extension = os.path.splitext(file_path)
        raise ValueError(f"Unsupported file format: '{extension}'. Please upload PDF, DOCX, or TXT.")
print("Helper functions defined.")


# --- DEEP LEARNING MODEL TRAINING PHASE ---
print("\n--- Starting Deep Learning Model Training Phase ---")

# Step 3: Prepare Internal Training Data
# !!! CRITICAL !!!
# !!! THIS IS YOUR MERGED DATASET !!!
data = {
    "text": [
        # Finance
        "The stock market closed higher today due to economic growth.",
        "Income tax planning is essential for all finance professionals.",
        "This document outlines quarterly business revenue projections.",
        "Investing in mutual funds is a safe financial strategy.",
        "The federal reserve announced new interest rate policies.",
        # Health
        "We are proud to offer healthcare benefits to all employees.",
        "The hospital reported an increase in patient satisfaction.",
        "Doctors recommend a healthy lifestyle for better well-being.",
        "Nutrition and exercise are key components of good health.",
        "A new study on vaccine efficacy has been published.",
        # Technology
        "AI is transforming industries with automation and data analytics.",
        "Quantum computing will change the future of machine learning.",
        "This software update includes security patches and UI improvements.",
        "The app integrates well with modern cloud platforms.",
        "Cybersecurity threats are becoming increasingly sophisticated.",
        # Language
        "Synonyms and antonyms are key to understanding English vocabulary.",
        "Common word meanings include definitions, translations, and opposites.",
        "The glossary lists over 500 advanced English words with Hindi meanings.",
        "Vocabulary building exercises improve language and communication skills.",
        "Word lists often include synonyms, antonyms, and contextual usage.",
        # Legal (New)
        "This contract outlines the terms and conditions of the agreement.",
        "The court ruled in favor of the plaintiff in the recent lawsuit.",
        "Legal compliance is mandatory for all business operations.",
        "Understanding intellectual property rights is crucial for innovators.",
        "The new legislation will impact data privacy regulations.",
        # Education (New)
        "The new curriculum focuses on project-based learning.",
        "University enrollment has increased this academic year.",
        "Online courses offer flexible learning opportunities for students.",
        "Teachers play a vital role in shaping future generations.",
        "Educational reforms aim to improve student outcomes.",
        # Sports (New)
        "The home team won the championship in a thrilling final match.",
        "Athlete training programs focus on strength and endurance.",
        "The Olympics showcase incredible sporting talent from around the world.",
        "Sports news covers scores, highlights, and player updates.",
        "The transfer window for major league soccer is now open.",
        # Politics (New)
        "The upcoming election will determine the new government leadership.",
        "Debates on foreign policy are ongoing in the parliament.",
        "Political campaigns are now actively using social media.",
        "Citizens' votes are essential for a democratic process.",
        "The government announced a new infrastructure spending bill."
    ],
    "category": [
        "finance", "finance", "finance", "finance", "finance",
        "health", "health", "health", "health", "health",
        "technology", "technology", "technology", "technology", "technology",
        "language", "language", "language", "language", "language",
        "legal", "legal", "legal", "legal", "legal",
        "education", "education", "education", "education", "education",
        "sports", "sports", "sports", "sports", "sports",
        "politics", "politics", "politics", "politics", "politics"
    ]
}
df = pd.DataFrame(data)
# --- END OF DATA LOADING ---

if len(df) < 50: # Arbitrary threshold, but good to warn
    print(f"   WARNING: The current dataset has only {len(df)} samples.")
    print("   Deep learning models generally require hundreds or thousands of samples per category to perform well.")
else:
    print(f"   Dataset has {len(df)} samples, which is a good starting point. More data per category is always better!")


print(f"   Internal data loaded: {len(df)} samples.")
if df.empty or 'category' not in df.columns:
    print("   ERROR: DataFrame is empty or 'category' column is missing. Exiting.")
    exit()

num_classes = df['category'].nunique()
unique_categories = sorted(df['category'].unique())
print(f"   Number of unique categories: {num_classes}")
print(f"   Categories: {unique_categories}")


# Step 4: Preprocess Data for Deep Learning
print("\nPreprocessing data for Deep Learning model...")
# Label Encoding
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])
y_categorical = to_categorical(df['category_encoded'], num_classes=num_classes)
print(f"   Labels one-hot encoded. Shape: {y_categorical.shape}")

# Tokenization and Sequencing
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>") # <unk> for unknown/out-of-vocabulary
tokenizer.fit_on_texts(df['text'])
X_sequences = tokenizer.texts_to_sequences(df['text'])

# Padding sequences
X_padded = pad_sequences(X_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
print(f"   Texts tokenized and padded. Shape: {X_padded.shape}")

# Store class names for later use
class_names = label_encoder.classes_
print(f"   Class names stored: {class_names}")

# Step 5: Split Data
X_train, X_test, y_train, y_test = None, None, None, None
test_data_available = False

# Ensure enough data for a meaningful split
if len(df) >= 2 * num_classes and len(df) > 10 : # Need at least 2 per class for stratification, and a reasonable total
    try:
        test_size = 0.2
        # Stratify if possible
        min_samples_per_class_for_split = 2 # For stratification, each class needs at least 2 samples to be in both train/test
        class_counts = df['category_encoded'].value_counts()

        can_stratify = all(count >= min_samples_per_class_for_split for count in class_counts) and \
                       int(len(df) * test_size) >= num_classes # Test set should be able to hold at least one of each class

        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_padded, y_categorical, test_size=test_size, random_state=42, stratify=df['category_encoded']
            )
            print(f"   Data split using stratification (test_size={test_size:.2f}).")
        else:
            print(f"   Warning: Cannot stratify effectively with current data distribution. Using regular split (test_size={test_size:.2f}).")
            X_train, X_test, y_train, y_test = train_test_split(
                X_padded, y_categorical, test_size=test_size, random_state=42
            )
        test_data_available = True
        print(f"   Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        print(f"   Error during data split: {e}. Training on all data.")
        X_train, y_train = X_padded, y_categorical
        test_data_available = False
else:
    print(f"   Dataset too small for a meaningful train/test split (Total: {len(df)}, Classes: {num_classes}). Training on all data.")
    X_train, y_train = X_padded, y_categorical
    test_data_available = False


# Step 6: Define and Compile Keras Model
print("\nDefining Keras model architecture...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
    # For simpler/faster models, especially with less data, consider:
    # GlobalMaxPooling1D(),
    # Or:
    # Conv1D(128, 5, activation='relu'),
    # GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 7: Train the Model
model_trained = False
if X_train is not None and y_train is not None and X_train.shape[0] > 0:
    print("\nTraining the model...")
    try:
        callbacks_list = []
        if test_data_available:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            callbacks_list.append(early_stopping)

        history = model.fit(
            X_train, y_train,
            epochs=50, # Increase epochs if you have more data
            batch_size=16 if len(X_train) < 100 else 32, # Smaller batch for smaller data
            validation_data=(X_test, y_test) if test_data_available else None,
            callbacks=callbacks_list,
            verbose=1
        )
        print("   Model training complete.")
        model_trained = True

        if history and hasattr(history, 'history') and 'accuracy' in history.history:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            if test_data_available and 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            if test_data_available and 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig("training_history.png")
            print("   Training history plot saved as training_history.png")

    except Exception as e:
        print(f"   Error during model training: {e}")
        model_trained = False
else:
    print("   Skipping training: No valid training data available or training data is empty.")


# Step 8: Evaluate Model
if model_trained and test_data_available and X_test is not None and y_test is not None and X_test.shape[0] > 0:
    print("\nEvaluating model on test set...")
    try:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test Loss: {loss:.4f}")
        print(f"   Test Accuracy: {accuracy:.4f}")

        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        print("\n   Classification Report (Test Set):")
        print(classification_report(y_test_classes, y_pred_classes, target_names=class_names, zero_division=0))
    except Exception as e:
        print(f"   Error during evaluation: {e}")
elif model_trained:
    print("\nSkipping evaluation (no test data available or test data is empty).")
else:
    print("\nSkipping evaluation (model not trained).")


# Step 9: Save the Trained Model and Tokenizer
keras_model_filename = 'document_classifier_dl_model.keras' # Recommended Keras format
tokenizer_filename = 'document_tokenizer.joblib'
class_names_filename = 'class_names.joblib'
model_assets_saved = False

if model_trained:
    print(f"\nSaving trained Keras model to {keras_model_filename}...")
    try:
        model.save(keras_model_filename)
        print(f"   Keras model saved successfully.")

        print(f"Saving tokenizer to {tokenizer_filename}...")
        joblib.dump(tokenizer, tokenizer_filename)
        print(f"   Tokenizer saved successfully.")

        print(f"Saving class names to {class_names_filename}...")
        joblib.dump(class_names, class_names_filename)
        print(f"   Class names saved successfully.")
        model_assets_saved = True
    except Exception as e:
        print(f"   Error saving model or tokenizer/class_names: {e}")
else:
    print("\nSkipping model asset saving (model not trained or training failed).")

print("--- Deep Learning Model Training Phase Finished ---")


# --- PREDICTION PHASE (using loaded DL model) ---
print("\n--- Starting Prediction Phase (Deep Learning Model) ---")

# Step 10: Load the Saved Model, Tokenizer, and Class Names
loaded_dl_model = None
loaded_tokenizer = None
loaded_class_names = None

if os.path.exists(keras_model_filename) and os.path.exists(tokenizer_filename) and os.path.exists(class_names_filename):
    print(f"Loading Keras model from {keras_model_filename}...")
    try:
        loaded_dl_model = load_model(keras_model_filename)
        print(f"   Keras model loaded successfully.")

        print(f"Loading tokenizer from {tokenizer_filename}...")
        loaded_tokenizer = joblib.load(tokenizer_filename)
        print(f"   Tokenizer loaded successfully.")

        print(f"Loading class names from {class_names_filename}...")
        loaded_class_names = joblib.load(class_names_filename)
        print(f"   Class names loaded successfully: {loaded_class_names}")

    except Exception as e:
        print(f"   Error loading model, tokenizer, or class names: {e}")
        loaded_dl_model = None
else:
    missing_files = []
    if not os.path.exists(keras_model_filename): missing_files.append(keras_model_filename)
    if not os.path.exists(tokenizer_filename): missing_files.append(tokenizer_filename)
    if not os.path.exists(class_names_filename): missing_files.append(class_names_filename)
    print(f"Cannot load model/tokenizer/class_names: One or more files not found: {', '.join(missing_files)}. Ensure training succeeded and files were saved.")


# Step 11: Upload Document using File Dialog
user_provided_filepath = None
extracted_text_for_dl = None
file_selected_dl = False

# CORRECTED Conditional Check (FIXED ValueError)
if loaded_dl_model is not None and loaded_tokenizer is not None and loaded_class_names is not None:
    print("\nPlease select a document (PDF, DOCX, or TXT) for DL classification using the file dialog...")
    try:
        root = Tk()
        root.withdraw()
        user_provided_filepath = filedialog.askopenfilename(
            title="Select Document for DL Classification",
            filetypes=(("Supported Files", "*.pdf *.docx *.txt"), ("All files", "*.*"))
        )
        root.destroy() # Close the hidden Tk window

        if user_provided_filepath:
            print(f"\n   File selected: {user_provided_filepath}")
            file_selected_dl = True
        else:
            print("   No file was selected (dialog cancelled).")
    except Exception as e:
        print(f"   An error occurred during file selection dialog: {e}")
        print("   Falling back to manual path input for DL model.")
        user_provided_filepath = input("\nEnter the full path to your document (PDF, DOCX, or TXT): ").strip()
        if user_provided_filepath and os.path.exists(user_provided_filepath):
             print(f"\n   File path received: {user_provided_filepath}")
             file_selected_dl = True
        elif user_provided_filepath:
             print(f"   Error: File not found at path: {user_provided_filepath}")
        else:
             print("   No file path was entered manually.")
else:
    print("\nSkipping file selection for DL: Model, tokenizer, or class names not loaded successfully.")


# Step 12: Extract Text from User's File
if file_selected_dl and user_provided_filepath:
    if not os.path.exists(user_provided_filepath):
         print(f"\n   Error: Selected file '{user_provided_filepath}' not found locally for extraction.")
    else:
        print(f"\nExtracting text from '{user_provided_filepath}' for DL model...")
        try:
            extracted_text_for_dl = extract_text(user_provided_filepath)
            print(f"   Text extracted successfully ({len(extracted_text_for_dl)} characters).")
        except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
            print(f"   Error during text extraction: {e}")
            extracted_text_for_dl = None


# Step 13: Preprocess and Predict Category using Loaded DL Model
if loaded_dl_model is not None and loaded_tokenizer is not None and loaded_class_names is not None and \
   extracted_text_for_dl is not None and len(extracted_text_for_dl.strip()) > 0:
    print(f"\nPredicting category for document '{os.path.basename(user_provided_filepath)}' using DL model...")
    try:
        new_sequences = loaded_tokenizer.texts_to_sequences([extracted_text_for_dl])
        new_padded = pad_sequences(new_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

        predictions_probs = loaded_dl_model.predict(new_padded, verbose=0)[0]
        predicted_class_index = np.argmax(predictions_probs)

        # Ensure predicted_class_index is within bounds of loaded_class_names
        if 0 <= predicted_class_index < len(loaded_class_names):
            predicted_category = loaded_class_names[predicted_class_index]
            predicted_confidence = predictions_probs[predicted_class_index]

            print(f"\n--- DEEP LEARNING FINAL PREDICTION ---")
            preview_text = extracted_text_for_dl[:150].replace(os.linesep, ' ').strip()
            print(f"   Document: '{os.path.basename(user_provided_filepath)}' (preview: '{preview_text}...')")
            print(f"   Predicted Category: >>> {predicted_category} <<< (Confidence: {predicted_confidence:.3f})")

            print("\n   Prediction Probabilities (DL Model):")
            prob_dict = dict(zip(loaded_class_names, predictions_probs))
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
            for cat, prob in sorted_probs.items():
                print(f"     - {cat}: {prob:.3f}")
        else:
            print(f"   Error: Predicted class index {predicted_class_index} is out of bounds for loaded class names (Count: {len(loaded_class_names)}).")

    except Exception as e:
        print(f"   An error occurred during DL prediction: {e}")
        import traceback
        traceback.print_exc()


elif loaded_dl_model is None or loaded_tokenizer is None or loaded_class_names is None:
    print("\nDL Prediction skipped: Model, tokenizer, or class names could not be loaded.")
elif not file_selected_dl:
     print("\nDL Prediction skipped: No file was selected or selection failed.")
elif extracted_text_for_dl is None or len(extracted_text_for_dl.strip()) == 0:
    print("\nDL Prediction skipped: Text could not be extracted or extracted text is empty.")
else:
     print("\nDL Prediction skipped due to an unknown issue after file selection/extraction for DL model.")

print("\n--- Full Process Finished ---")