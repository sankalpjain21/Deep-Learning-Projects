#!/usr/bin/env python3
import sys
import textwrap
import heapq
from string import punctuation
import os

# --- Configuration ---
CONFIG = {
    "nltk": {
        "default_summary_sentences": 3,
        "resources_to_check": {
            "punkt": "tokenizers/punkt",
            "stopwords": "corpora/stopwords",
            "punkt_tab": "tokenizers/punkt_tab"
        }
    },
    "transformers": {
        # Common models: "t5-small", "t5-base", "facebook/bart-large-cnn", "google/pegasus-xsum"
        "default_model_name": "t5-small", # "t5-small" is quick, "facebook/bart-large-cnn" is better but larger
        "default_min_output_length_factor": 0.15, # Percentage of input words (approx)
        "default_max_output_length_factor": 0.4,  # Percentage of input words (approx)
        "min_absolute_output_length": 20,         # Minimum tokens in summary
        "max_absolute_output_length": 150,        # Maximum tokens (for t5-small, BART can handle more)
        "device": -1,  # -1 for CPU, 0 for first CUDA GPU, etc.
    },
    "general": {
        "preview_length": 500,
        "output_width": 80
    }
}

# --- Library Availability Flags ---
NLTK_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

# --- Attempt NLTK Imports ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("INFO: NLTK library not found. Frequency-based summarization (Method 1) will be unavailable.")
    print("      Install it using: pip install nltk")

# --- Attempt Transformers Imports ---
try:
    import transformers
    try:
        import torch
    except ImportError:
        try:
            import tensorflow
        except ImportError:
            raise ImportError("Neither PyTorch nor TensorFlow backend found for Transformers.")
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"INFO: Transformers library or its backend not found: {e}")
    print("Abstractive summarization (Method 2) will be unavailable.")
    print("Install necessary packages, e.g.: pip install transformers torch sentencepiece")
    print("or pip install transformers tensorflow sentencepiece if using TensorFlow")


# ===== UTILITY FUNCTIONS =====
def download_nltk_data_if_needed():
    """Checks for and downloads required NLTK data."""
    if not NLTK_AVAILABLE:
        return False

    data_downloaded_successfully = True
    resources = CONFIG["nltk"]["resources_to_check"]

    print("Checking NLTK resources...")
    for name, path_fragment in resources.items():
        try:
            nltk.data.find(path_fragment)
        except LookupError:
            print(f"INFO: NLTK resource '{name}' not found. Attempting download...")
            try:
                nltk.download(name, quiet=True)
                print(f"NLTK resource '{name}' downloaded successfully.")
            except Exception as download_error:
                print(f"\n*** WARNING: Failed to download NLTK resource '{name}': {download_error} ***")
                print(f"*** Please try manual download: In Python, run 'import nltk; nltk.download(\"{name}\")' ***")
                data_downloaded_successfully = False
        except Exception as check_error:
            print(f"\n*** WARNING: Error during NLTK data check for '{name}': {check_error} ***")
            data_downloaded_successfully = False

    if data_downloaded_successfully:
        print("NLTK data check complete.")
    else:
        print("NLTK data check encountered issues. NLTK-based summarization might be affected.")
    return data_downloaded_successfully

# ===== MODEL LOADING & SUMMARIZATION LOGIC =====

# --- Method 1: NLTK (Extractive) ---
def summarize_extractive_nltk(text, requested_num_sentences):
    """
    Generates an extractive summary using NLTK frequency scoring.
    Caps requested_num_sentences to the number of available sentences.
    """
    if not NLTK_AVAILABLE:
        return "Error: NLTK library is not available for extractive summarization."

    try:
        original_sentences = sent_tokenize(text)
        num_available_sentences = len(original_sentences)

        if num_available_sentences == 0:
             return "Info: No sentences found in the input text to summarize."
        if num_available_sentences < 2 and requested_num_sentences > 1 : # Not much to summarize from 1 sentence if more are asked
            print(f"Info: Only {num_available_sentences} sentence available. Returning it as summary.")
            return " ".join(original_sentences)


        # Cap the requested_num_sentences
        num_sentences_to_extract = min(requested_num_sentences, num_available_sentences)
        
        if requested_num_sentences > num_available_sentences and num_available_sentences > 0 :
             print(f"Info: Requested {requested_num_sentences} sentences, but only {num_available_sentences} are available. Will extract up to {num_available_sentences} sentences.")
        
        stop_words = set(stopwords.words('english') + list(punctuation))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word not in stop_words and word.isalnum()]

        if not filtered_words:
            # If original text had sentences but all words were filtered out
            if num_available_sentences > 0:
                 return "Info: No meaningful (non-stopword, alphanumeric) words found to score sentences. Cannot create NLTK summary."
            return "Info: No meaningful words found after filtering. Cannot create NLTK summary."


        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / max_frequency)

        sentence_scores = {}
        for i, sentence in enumerate(original_sentences):
            score = 0
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    score += word_frequencies[word]
            if score > 0:
                sentence_scores[i] = score
        
        if not sentence_scores:
            if num_sentences_to_extract == num_available_sentences and num_available_sentences > 0:
                return " ".join(original_sentences)
            return "Info: No sentences received a score high enough to be included in the summary."

        actual_num_to_extract_from_scored = min(num_sentences_to_extract, len(sentence_scores))
        
        if actual_num_to_extract_from_scored <= 0 :
             if num_available_sentences > 0 and num_sentences_to_extract == num_available_sentences and len(sentence_scores) == 0:
                 return " ".join(original_sentences)
             return "Info: Could not extract any scored sentences for the summary."

        top_sentence_indices = heapq.nlargest(actual_num_to_extract_from_scored, sentence_scores, key=sentence_scores.get)
        top_sentence_indices.sort()

        summary_sentences = [original_sentences[i] for i in top_sentence_indices]
        return " ".join(summary_sentences)

    except Exception as e:
        import traceback
        return f"Error during NLTK summarization ({type(e).__name__}): {e}\n{traceback.format_exc(limit=1)}"

# --- Method 2: Transformers (Abstractive - Deep Learning) ---
def load_transformer_summarizer(model_name, device):
    """Loads a Hugging Face summarization pipeline."""
    if not TRANSFORMERS_AVAILABLE:
        return None, "Error: Transformers library not available for abstractive summarization."
    try:
        print(f"\nLoading abstractive summarization model: '{model_name}'...")
        print(f"This may take time on first download. Using device: {'GPU '+str(device) if device >=0 else 'CPU'}")
        
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
        print(f"Model '{model_name}' loaded successfully.")
        return summarizer, None
    except Exception as e:
        import traceback
        return None, f"Error loading Transformers model '{model_name}' ({type(e).__name__}): {e}\n{traceback.format_exc(limit=1)}"

def summarize_abstractive_transformer(summarizer_pipeline, text, min_length, max_length):
    """Generates an abstractive summary using a loaded Transformer pipeline."""
    if summarizer_pipeline is None:
        return "Error: Summarizer pipeline not loaded."
    try:
        print(f"Generating abstractive summary (target length: {min_length}-{max_length} tokens)...")
        summary_outputs = summarizer_pipeline(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
            truncation=True # Important for handling long inputs
        )
        if summary_outputs and isinstance(summary_outputs, list) and summary_outputs[0] and 'summary_text' in summary_outputs[0]:
            return summary_outputs[0]['summary_text']
        else:
            return f"Error: Unexpected output format from summarization model: {summary_outputs}"
    except Exception as e:
        import traceback
        return f"Error during Transformers summarization ({type(e).__name__}): {e}\n{traceback.format_exc(limit=1)}"

# ===== MAIN APPLICATION LOGIC =====
def main():
    print("=" * CONFIG["general"]["output_width"])
    print("   Text Summarizer") # Simplified title
    print("=" * CONFIG["general"]["output_width"])

    if not NLTK_AVAILABLE and not TRANSFORMERS_AVAILABLE:
        print("\nERROR: Neither NLTK nor Transformers libraries are available. Cannot proceed.")
        print("Please install requirements. Example: 'pip install nltk transformers torch sentencepiece'")
        sys.exit(1)

    nltk_ready = download_nltk_data_if_needed() if NLTK_AVAILABLE else False

    print("\nEnter or paste your text. Press Enter on an empty line (after typing something) to finish:")
    user_input_lines = []
    while True:
        try:
            line = input()
            if not line.strip() and user_input_lines: 
                break
            if not line.strip() and not user_input_lines:
                print("(Type your text or press Enter again if you intend to submit empty text)")
                continue
            user_input_lines.append(line)
        except EOFError:
            print("\n(EOF detected, processing input...)")
            break
    print("--- End Text Input ---")
    original_text = " ".join(user_input_lines).strip()

    if not original_text:
        print("\nNo text provided. Exiting.")
        sys.exit(0)

    print("\n--- Original Text (Preview) ---")
    preview = original_text[:CONFIG["general"]["preview_length"]] + \
              ("..." if len(original_text) > CONFIG["general"]["preview_length"] else "")
    print(textwrap.fill(preview, width=CONFIG["general"]["output_width"]))
    print("-" * CONFIG["general"]["output_width"])

    available_methods = {}
    print("\nChoose a summarization method:")
    if NLTK_AVAILABLE and nltk_ready:
        print("  1: Extractive (NLTK Frequency-Based - Fast, uses original sentences)")
        available_methods["1"] = "nltk"
    elif NLTK_AVAILABLE:
         print("  (NLTK method might be unreliable due to NLTK data issues)")

    if TRANSFORMERS_AVAILABLE:
        print(f"  2: Abstractive (AI Transformers - Model: {CONFIG['transformers']['default_model_name']})")
        available_methods["2"] = "transformers"

    if not available_methods:
         print("\nCritical Error: No summarization methods are available. Exiting.")
         sys.exit(1)

    method_choice = ""
    while method_choice not in available_methods:
        method_choice = input(f"Enter your choice ({'/'.join(available_methods.keys())}): ").strip()

    chosen_method_type = available_methods[method_choice]
    summary_text = ""

    print(f"\n--- Generating Summary using {chosen_method_type.upper()} ---")

    if chosen_method_type == "nltk":
        num_available_original_sents = 0
        if NLTK_AVAILABLE and nltk_ready:
            try:
                num_available_original_sents = len(sent_tokenize(original_text))
            except Exception:
                num_available_original_sents = len(original_text.split('. ')) 
        
        default_sents_to_show = 1
        if num_available_original_sents > 0:
            default_sents_to_show = max(1, num_available_original_sents // 3 if num_available_original_sents > 3 else 1) # Adjusted default
        
        while True:
            try:
                prompt_msg = f"Enter desired number of sentences for NLTK summary [max: {num_available_original_sents if num_available_original_sents > 0 else 1}, default: {default_sents_to_show}]: "
                if num_available_original_sents == 0:
                     prompt_msg = f"Enter desired number of sentences for NLTK summary [default: {default_sents_to_show}]: "

                num_sents_str = input(prompt_msg)
                requested_sents = int(num_sents_str) if num_sents_str else default_sents_to_show
                
                if requested_sents > 0:
                    break
                else:
                    print("Please enter a positive number for sentences.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        summary_text = summarize_extractive_nltk(original_text, requested_sents)

    elif chosen_method_type == "transformers":
        transformer_model_name = CONFIG["transformers"]["default_model_name"]
        summarizer_pipeline, error_msg = load_transformer_summarizer(
            transformer_model_name,
            CONFIG["transformers"]["device"]
        )
        if error_msg:
            summary_text = error_msg
        else:
            num_words_estimate = len(original_text.split())
            min_len = max(CONFIG["transformers"]["min_absolute_output_length"],
                          int(num_words_estimate * CONFIG["transformers"]["default_min_output_length_factor"]))
            max_len = min(CONFIG["transformers"]["max_absolute_output_length"],
                          int(num_words_estimate * CONFIG["transformers"]["default_max_output_length_factor"]))
            
            # Ensure max_len is reasonably larger than min_len and respects model constraints
            max_len = max(max_len, min_len + 15) # e.g. min_len + 15 or 20
            if "t5" in transformer_model_name: # t5-small max sequence length is 512
                max_len = min(max_len, 480) # Leave some buffer from model's absolute max
            elif "bart" in transformer_model_name.lower(): # bart-large max sequence length is 1024
                max_len = min(max_len, 900) # Leave some buffer

            # Ensure min_len is not greater than max_len after adjustments
            min_len = min(min_len, max_len - 10 if max_len > 10 else max_len)


            summary_text = summarize_abstractive_transformer(summarizer_pipeline, original_text, min_len, max_len)

    # --- Print Summary ---
    print("\n" + "=" * CONFIG["general"]["output_width"])
    print(f"         SUMMARY ({chosen_method_type.upper()})")
    print("=" * CONFIG["general"]["output_width"])

    if summary_text is None: 
        summary_text = "" 

    if summary_text.startswith("Error:") or \
       summary_text.startswith("Info:") or \
       summary_text.startswith("Warning:"):
        print(summary_text)
    elif not summary_text: 
         print("No summary could be generated or an unexpected issue occurred.")
    else:
        print(textwrap.fill(summary_text, width=CONFIG["general"]["output_width"]))
    print("=" * CONFIG["general"]["output_width"])

    # Optional: Keep or remove the "Further Considerations" based on your preference
    # print("\n--- Further Deep Learning Project Considerations ---")
    # print("For a full DL project, you would typically add:")
    # print("- Data loading/preprocessing pipelines (for training/fine-tuning).")
    # ...

if __name__ == "__main__":
    main()