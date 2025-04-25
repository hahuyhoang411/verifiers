"""
Processes and analyzes categorized tool usage datasets, prepares train/test splits,
and uploads them to the Hugging Face Hub.

Assumes the following prerequisite steps have been completed:
1. SGLang server is running:
   python -m sglang.launch_server \
     --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
     --port 30000 \
     --host 0.0.0.0 \
     --dp 8

2. Categorization script has been run to generate the input JSONL files:
   python sglang_category_tool.py \
     --dataset-name "HoangHa/Tool-RL" \
     --dataset-sub "train-raw" \
     --output-file "instruct_tool_categorized.jsonl" \ # Or your instruct path
     --prompt-column "question" \
     --uuid-column "question" \
     --api-addr "127.0.0.1:30000" \
     --num-generations 1 # Example: Update if needed
     --max-tokens 16384 \
     --max-concurrent 200 \
     --max-retries 3

   python sglang_category_tool.py \
     --dataset-name "HoangHa/Tool-RL" \
     --dataset-sub "train-raw" \
     --output-file "distil_tool_categorized.jsonl" \ # Or your distil path
     --prompt-column "question" \
     --uuid-column "question" \
     --api-addr "127.0.0.1:30000" \
     --num-generations 2 # Example: Update if needed
     --max-tokens 16384 \
     --max-concurrent 200 \
     --max-retries 3

Note: Adjust paths and parameters in the commands above as needed.
The Hugging Face token should be handled securely (e.g., environment variable)
instead of being hardcoded.
"""

import os
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi, HfFolder # For token management

# --- Configuration ---
# Replace with your actual paths and settings
INSTRUCT_CATEGORIZED_FILE = "instruct_tool_categorized.jsonl"
DISTIL_CATEGORIZED_FILE = "distil_tool_categorized.jsonl"
HF_REPO_ID = "HoangHa/Tool-RL"
HF_TOKEN = "" # !!! IMPORTANT: Replace with your token or use HF login/env var !!!
# Consider using: HF_TOKEN = os.environ.get("HF_TOKEN") or HfFolder.get_token()

# Splitting parameters
TARGET_EVAL_SAMPLES_PER_CAT = 50
RANDOM_SEED = 42

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def preprocess_generations(example, generation_col_name, new_col_name):
    """Safely extracts 'generations' into a new column, handling missing/empty cases."""
    generations = example.get(generation_col_name)
    # Ensure it's always a list of strings, even if empty or None
    if isinstance(generations, list) and all(isinstance(g, str) for g in generations):
        example[new_col_name] = generations
    else:
        example[new_col_name] = [] # Default to empty list
    return example

def count_correct_simple(example, answer_col='answer', generated_col='new_answers', output_col='correct_new_answers_count'):
    """
    Compares 'answer' with each item in 'generated_col'.
    Counts how many generated answers contain the original answer (case-insensitive substring).
    """
    original_answer = example.get(answer_col)
    generated_answers_list = example.get(generated_col)
    count = 0

    if not isinstance(original_answer, str) or not isinstance(generated_answers_list, list):
        return {output_col: 0}

    processed_original_answer = original_answer.strip().lower()
    if not processed_original_answer:
         return {output_col: 0}

    for gen_answer in generated_answers_list:
        if isinstance(gen_answer, str):
            processed_gen_answer = gen_answer.strip().lower()
            if processed_original_answer in processed_gen_answer:
                count += 1

    return {output_col: count}

def count_correct_after_think(example, answer_col='answer', generated_col='new_answers_dis', output_col='correct_after_think_count'):
    """
    Counts how many strings in 'generated_col' contain the 'answer'
    *after* the first '</think>' marker (case-insensitive substring check).
    """
    original_answer = example.get(answer_col)
    generated_answers_list = example.get(generated_col)
    count = 0
    marker = '</think>'

    if not isinstance(original_answer, str) or not original_answer.strip():
        return {output_col: 0}

    processed_original_answer = original_answer.strip().lower()

    if not isinstance(generated_answers_list, list):
         return {output_col: 0} # Should be list after preprocessing

    for gen_answer_str in generated_answers_list:
        if not isinstance(gen_answer_str, str):
            continue # Skip non-string items

        marker_pos = gen_answer_str.find(marker)
        if marker_pos != -1:
            text_after_marker = gen_answer_str[marker_pos + len(marker):]
            processed_gen_answer_part = text_after_marker.strip().lower()
            if processed_original_answer in processed_gen_answer_part:
                count += 1

    return {output_col: count}

def add_dis_count_by_id(example, lookup, id_col='id', count_col='correct_after_think_count'):
    """Adds count from the lookup dictionary based on ID."""
    example_id = example.get(id_col)
    # Default to -1 if ID not found, to allow filtering later
    example[count_col] = lookup.get(example_id, -1)
    return example

def plot_agreement_distribution(df, x_col, y_col, title, xlabel, ylabel, order=None):
    """Generates and displays a bar plot for agreement categories."""
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x_col, y=y_col, data=df, palette='viridis', order=order)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add counts on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=10)

    plt.tight_layout()
    plt.show()
    logging.info("Plot displayed.")

def convert_research_str_to_bool(example):
    """Maps 'Yes'/'No' string in 'research_str' to boolean in 'research'."""
    if 'research_str' not in example:
        logging.warning(f"Missing 'research_str' in example: {example.get('id', 'Unknown ID')}")
        return {'research': None} # Or handle as error
    return {'research': True if example['research_str'] == 'Yes' else False}

def push_to_hub_safe(dataset, repo_id, split_name, token):
    """Pushes a dataset to the hub with logging."""
    if dataset is None or len(dataset) == 0:
        logging.warning(f"Skipping push for empty dataset: {split_name}")
        return
    try:
        logging.info(f"Pushing dataset '{split_name}' ({len(dataset)} rows) to {repo_id}...")
        dataset.push_to_hub(repo_id, split_name, token=token)
        logging.info(f"Successfully pushed '{split_name}' to {repo_id}.")
    except Exception as e:
        logging.error(f"Failed to push dataset '{split_name}' to {repo_id}: {e}")
        # Decide if you want to raise the exception or just log it
        # raise e

# --- Main Execution ---
if __name__ == "__main__":

    # --- Step 1: Load and Process "Instruct" Dataset ---
    logging.info(f"--- Loading and Processing Instruct Dataset: {INSTRUCT_CATEGORIZED_FILE} ---")
    try:
        dataset_ins = load_dataset("json", data_files=INSTRUCT_CATEGORIZED_FILE, split='train')
        logging.info(f"Loaded instruct dataset with {len(dataset_ins)} rows.")

        # Preprocess 'generations' column safely
        dataset_ins = dataset_ins.map(
            preprocess_generations,
            fn_kwargs={'generation_col_name': 'generations', 'new_col_name': 'new_answers'}
        )
        dataset_ins = dataset_ins.remove_columns(['generations', 'finish_reasons', 'api_metadata'])
        logging.info(f"Instruct dataset preprocessed. Columns: {dataset_ins.column_names}")
        # print(dataset_ins[0]) # Optional: Inspect first example

        # Calculate simple correctness
        dataset_ins_updated = dataset_ins.map(count_correct_simple)
        logging.info("Calculated simple correctness count for instruct dataset.")
        # print(dataset_ins_updated[0]) # Optional: Inspect first example with count

        # Calculate and log overall accuracy (optional)
        total_new_answers = sum(len(na_list) for na_list in dataset_ins_updated['new_answers'])
        total_correct_new_answers = sum(dataset_ins_updated['correct_new_answers_count'])
        if total_new_answers > 0:
            overall_accuracy = (total_correct_new_answers / total_new_answers) * 100
            logging.info(f"Instruct Dataset - Overall Accuracy (Simple Substring Check): {overall_accuracy:.2f}% "
                         f"({total_correct_new_answers}/{total_new_answers})")
        else:
            logging.info("Instruct Dataset - No 'new_answers' found to calculate accuracy.")

    except FileNotFoundError:
        logging.error(f"Instruct dataset file not found: {INSTRUCT_CATEGORIZED_FILE}. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Error processing instruct dataset: {e}")
        exit(1)

    # --- Step 2: Load and Process "Distil" Dataset ---
    logging.info(f"--- Loading and Processing Distil Dataset: {DISTIL_CATEGORIZED_FILE} ---")
    try:
        dataset_dis = load_dataset("json", data_files=DISTIL_CATEGORIZED_FILE, split='train')
        logging.info(f"Loaded distil dataset with {len(dataset_dis)} rows.")

        # Preprocess 'generations' column safely
        dataset_dis = dataset_dis.map(
            preprocess_generations,
            fn_kwargs={'generation_col_name': 'generations', 'new_col_name': 'new_answers_dis'}
        )
        dataset_dis = dataset_dis.remove_columns(['generations', 'finish_reasons', 'api_metadata'])
        logging.info(f"Distil dataset preprocessed. Columns: {dataset_dis.column_names}")
        # print(dataset_dis[0]) # Optional: Inspect first example

        # Calculate correctness after '</think>'
        dataset_dis_updated = dataset_dis.map(count_correct_after_think)
        logging.info("Calculated 'after think' correctness count for distil dataset.")
        # print(dataset_dis_updated[0]) # Optional: Inspect first example with count

        # Calculate and log overall accuracy after '</think>' (optional)
        output_key = 'correct_after_think_count'
        marker = '</think>'
        total_correct_found_dis = 0
        total_potential_answers_checked = 0
        for example in dataset_dis_updated:
            total_correct_found_dis += example[output_key]
            new_answers_list = example.get('new_answers_dis', [])
            if isinstance(new_answers_list, list):
                for new_answer_str in new_answers_list:
                     if isinstance(new_answer_str, str) and marker in new_answer_str:
                         total_potential_answers_checked += 1

        logging.info(f"Distil Dataset - Total Examples: {len(dataset_dis_updated)}")
        logging.info(f"Distil Dataset - Total 'new_answers_dis' containing '{marker}': {total_potential_answers_checked}")
        logging.info(f"Distil Dataset - Total Correct Answers found (after '{marker}'): {total_correct_found_dis}")
        if total_potential_answers_checked > 0:
            accuracy_dis = (total_correct_found_dis / total_potential_answers_checked) * 100
            logging.info(f"Distil Dataset - Accuracy (Correct / Total with Marker): {accuracy_dis:.2f}%")
        else:
            logging.info(f"Distil Dataset - No answers containing '{marker}' found. Accuracy cannot be calculated.")

    except FileNotFoundError:
        logging.error(f"Distil dataset file not found: {DISTIL_CATEGORIZED_FILE}. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Error processing distil dataset: {e}")
        exit(1)

    # --- Step 3: Combine Counts using ID ---
    logging.info("--- Combining Correctness Counts ---")
    if 'id' not in dataset_ins_updated.column_names or 'id' not in dataset_dis_updated.column_names:
         logging.error("Both datasets must have an 'id' column for combining. Exiting.")
         exit(1)

    # Create a lookup dictionary {id: count} from the distil dataset
    dis_counts_lookup = {example['id']: example['correct_after_think_count'] for example in dataset_dis_updated}
    logging.info(f"Created lookup dictionary with {len(dis_counts_lookup)} entries from distil dataset.")

    # Add the distil count to the instruct dataset
    dataset_combined = dataset_ins_updated.map(add_dis_count_by_id, fn_kwargs={'lookup': dis_counts_lookup})

    # Filter out rows where the ID wasn't found in the distil dataset (count is -1)
    initial_combined_len = len(dataset_combined)
    dataset_combined = dataset_combined.filter(lambda x: x['correct_after_think_count'] != -1)
    filtered_count = initial_combined_len - len(dataset_combined)
    logging.info(f"Combined dataset created. Initial rows: {initial_combined_len}. Rows after filtering missing IDs: {len(dataset_combined)}. ({filtered_count} rows removed)")

    # --- Step 4: Filter into Agreement Categories ---
    logging.info("--- Filtering into Agreement Categories ---")
    agreement_categories = {}
    try:
        agreement_categories["both_0"] = dataset_combined.filter(
            lambda x: x['correct_new_answers_count'] == 0 and x['correct_after_think_count'] == 0
        )
        agreement_categories["both_1"] = dataset_combined.filter(
            lambda x: x['correct_new_answers_count'] == 1 and x['correct_after_think_count'] == 1
        )
        agreement_categories["both_2"] = dataset_combined.filter(
            lambda x: x['correct_new_answers_count'] == 2 and x['correct_after_think_count'] == 2
        )
        agreement_categories["both_3plus"] = dataset_combined.filter(
            lambda x: x['correct_new_answers_count'] >= 3 and x['correct_after_think_count'] >= 3
        )

        logging.info("Filtering complete. Counts per agreement category:")
        category_labels = {
            "both_0": "Both 0 Correct", "both_1": "Both 1 Correct",
            "both_2": "Both 2 Correct", "both_3plus": "Both 3+ Correct"
        }
        agreement_counts = {cat_label: len(ds) for cat_key, ds in agreement_categories.items()
                            for cat_label in [category_labels[cat_key]]} # Get counts for plot
        for label, count in agreement_counts.items():
             logging.info(f"  {label}: {count}")

        total_filtered_rows = sum(agreement_counts.values())
        logging.info(f"Total rows across filtered agreement datasets: {total_filtered_rows}")
        if total_filtered_rows != len(dataset_combined):
            logging.warning(f"Mismatch between combined dataset size ({len(dataset_combined)}) and sum of agreement categories ({total_filtered_rows}). "
                            "This indicates disagreement cases were excluded.")

    except Exception as e:
        logging.error(f"Error during filtering into agreement categories: {e}")
        exit(1)

    # --- Step 5: Visualize Agreement Distribution ---
    logging.info("--- Generating Agreement Distribution Plot ---")
    try:
        agreement_data = {
            'Agreement Category': list(agreement_counts.keys()),
            'Number of Samples': list(agreement_counts.values())
        }
        agreement_df = pd.DataFrame(agreement_data)
        plot_order = ["Both 0 Correct", "Both 1 Correct", "Both 2 Correct", "Both 3+ Correct"]

        plot_agreement_distribution(
            df=agreement_df,
            x_col='Agreement Category',
            y_col='Number of Samples',
            title='Distribution of Samples where Both Correctness Counts Agree',
            xlabel='Agreement Category',
            ylabel='Number of Samples',
            order=plot_order
        )
        print("\nSummary Statistics of Agreement Counts:") # Keep print for table format
        print(agreement_df.set_index('Agreement Category'))

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        # Continue execution even if plotting fails

    # --- Step 6: Push Intermediate Filtered Datasets to Hub (Optional) ---
    # Consider whether these intermediate pushes are necessary for your workflow.
    logging.info("--- Pushing Intermediate Filtered Datasets to Hub ---")
    if not HF_TOKEN:
        logging.warning("HF_TOKEN not set. Skipping push of intermediate datasets.")
    else:
        push_to_hub_safe(agreement_categories["both_0"], HF_REPO_ID, "0correct", HF_TOKEN)
        push_to_hub_safe(agreement_categories["both_1"], HF_REPO_ID, "1correct", HF_TOKEN)
        push_to_hub_safe(agreement_categories["both_2"], HF_REPO_ID, "2correct", HF_TOKEN)
        push_to_hub_safe(agreement_categories["both_3plus"], HF_REPO_ID, "3correct", HF_TOKEN)


    # --- Step 7: Prepare Balanced Train/Evaluation Splits from 0 and 3+ Categories ---
    logging.info("--- Preparing Train/Evaluation Splits ---")
    dataset_0_correct = agreement_categories["both_0"]
    dataset_3plus_correct = agreement_categories["both_3plus"]
    initial_len_0 = len(dataset_0_correct)
    initial_len_3plus = len(dataset_3plus_correct)
    logging.info(f"Initial sizes for splitting: 0 Correct={initial_len_0}, 3+ Correct={initial_len_3plus}")

    evaluation_dataset = None
    research_dataset = None # This will be the final training set

    if initial_len_0 == 0 or initial_len_3plus == 0:
        logging.warning("Cannot create balanced train/eval splits as at least one category (0 or 3+) is empty.")
    else:
        # Determine Evaluation Set Size
        eval_samples_per_cat = min(TARGET_EVAL_SAMPLES_PER_CAT, initial_len_0, initial_len_3plus)
        logging.info(f"Target evaluation samples per category: {TARGET_EVAL_SAMPLES_PER_CAT}")
        if eval_samples_per_cat < TARGET_EVAL_SAMPLES_PER_CAT:
            logging.warning(f"Using {eval_samples_per_cat} samples per category for evaluation due to data limits.")
        else:
            logging.info(f"Selecting {eval_samples_per_cat} samples per category for evaluation set.")

        # Shuffle Original Datasets
        shuffled_0 = dataset_0_correct.shuffle(seed=RANDOM_SEED)
        shuffled_3plus = dataset_3plus_correct.shuffle(seed=RANDOM_SEED)

        # Create Evaluation Set (with temporary string label)
        selected_eval_sets = []
        if eval_samples_per_cat > 0:
            eval_0 = shuffled_0.select(range(eval_samples_per_cat))
            eval_0_with_str = eval_0.add_column(name="research_str", column=['Yes'] * eval_samples_per_cat) # Yes = Needs research (0 correct)
            selected_eval_sets.append(eval_0_with_str)

            eval_3plus = shuffled_3plus.select(range(eval_samples_per_cat))
            eval_3plus_with_str = eval_3plus.add_column(name="research_str", column=['No'] * eval_samples_per_cat) # No = Doesn't need research (3+ correct)
            selected_eval_sets.append(eval_3plus_with_str)

            evaluation_dataset_temp = concatenate_datasets(selected_eval_sets).shuffle(seed=RANDOM_SEED)
            logging.info(f"Created temporary evaluation set with {len(evaluation_dataset_temp)} rows ({eval_samples_per_cat} from each category).")

            # Convert 'research_str' to boolean 'research'
            logging.info("Converting 'research' column to boolean for evaluation dataset...")
            evaluation_dataset = evaluation_dataset_temp.map(convert_research_str_to_bool, remove_columns=['research_str'])
            logging.info(f"Evaluation dataset created. Features: {evaluation_dataset.features}")

        else:
            logging.warning("Evaluation set size is 0. Skipping creation.")

        # Prepare Remaining Data for Training
        remaining_indices_0 = range(eval_samples_per_cat, initial_len_0)
        remaining_indices_3plus = range(eval_samples_per_cat, initial_len_3plus)

        remaining_0 = shuffled_0.select(remaining_indices_0) if len(remaining_indices_0) > 0 else Dataset.from_dict(shuffled_0.features.copy())
        remaining_3plus = shuffled_3plus.select(remaining_indices_3plus) if len(remaining_indices_3plus) > 0 else Dataset.from_dict(shuffled_3plus.features.copy())
        logging.info(f"Remaining samples for potential training: {len(remaining_0)} (0-correct), {len(remaining_3plus)} (3+ correct)")


        # Determine Training Set Size (Balanced)
        train_samples_per_cat = min(len(remaining_0), len(remaining_3plus))
        logging.info(f"Selecting {train_samples_per_cat} samples per category for the balanced training set.")

        # Create Training Set (with temporary string label)
        selected_train_sets = []
        if train_samples_per_cat > 0:
            train_0 = remaining_0.select(range(train_samples_per_cat))
            train_0_with_str = train_0.add_column(name="research_str", column=['Yes'] * train_samples_per_cat)
            selected_train_sets.append(train_0_with_str)

            train_3plus = remaining_3plus.select(range(train_samples_per_cat))
            train_3plus_with_str = train_3plus.add_column(name="research_str", column=['No'] * train_samples_per_cat)
            selected_train_sets.append(train_3plus_with_str)

            research_dataset_temp = concatenate_datasets(selected_train_sets).shuffle(seed=RANDOM_SEED)
            logging.info(f"Created temporary training set with {len(research_dataset_temp)} rows ({train_samples_per_cat} from each category).")

            # Convert 'research_str' to boolean 'research'
            logging.info("Converting 'research' column to boolean for training dataset...")
            research_dataset = research_dataset_temp.map(convert_research_str_to_bool, remove_columns=['research_str'])
            logging.info(f"Training dataset (research_dataset) created. Features: {research_dataset.features}")

        else:
            logging.warning("Training set size is 0 (due to insufficient remaining data after eval split). Skipping creation.")

    # --- Step 8: Final Cleanup and Verification ---
    logging.info("--- Final Cleanup and Verification ---")
    cols_to_remove = ['new_answers', 'new_answers_dis', 'correct_new_answers_count', 'correct_after_think_count']

    if research_dataset:
        # Make sure columns exist before trying to remove
        actual_cols_to_remove_train = [col for col in cols_to_remove if col in research_dataset.column_names]
        if actual_cols_to_remove_train:
            research_dataset = research_dataset.remove_columns(actual_cols_to_remove_train)
            logging.info(f"Removed intermediate columns from training dataset. Final columns: {research_dataset.column_names}")
        else:
            logging.info("No intermediate columns to remove from training dataset.")

        train_bool_values = np.array(research_dataset['research'])
        true_count_train = np.sum(train_bool_values)
        false_count_train = len(train_bool_values) - true_count_train
        logging.info(f"Final Training Dataset Size: {len(research_dataset)}")
        logging.info(f"  Train 'research' counts: True={true_count_train} (Needs Research/0-Correct), False={false_count_train} (No Research/3+ Correct)")

    if evaluation_dataset:
        # Make sure columns exist before trying to remove
        actual_cols_to_remove_eval = [col for col in cols_to_remove if col in evaluation_dataset.column_names]
        if actual_cols_to_remove_eval:
            evaluation_dataset = evaluation_dataset.remove_columns(actual_cols_to_remove_eval)
            logging.info(f"Removed intermediate columns from evaluation dataset. Final columns: {evaluation_dataset.column_names}")
        else:
             logging.info("No intermediate columns to remove from evaluation dataset.")

        eval_bool_values = np.array(evaluation_dataset['research'])
        true_count_eval = np.sum(eval_bool_values)
        false_count_eval = len(eval_bool_values) - true_count_eval
        logging.info(f"Final Evaluation Dataset Size: {len(evaluation_dataset)}")
        logging.info(f"  Eval 'research' counts: True={true_count_eval} (Needs Research/0-Correct), False={false_count_eval} (No Research/3+ Correct)")


    # --- Step 9: Push Final Train/Test Datasets to Hub ---
    logging.info("--- Pushing Final Train/Test Datasets to Hub ---")
    if not HF_TOKEN:
        logging.warning("HF_TOKEN not set. Skipping final push of train/test datasets.")
    else:
        # Use 'test' split name for evaluation dataset on the hub
        push_to_hub_safe(research_dataset, HF_REPO_ID, "train", HF_TOKEN)
        push_to_hub_safe(evaluation_dataset, HF_REPO_ID, "test", HF_TOKEN)

    logging.info("--- Script execution finished ---")