from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np

dataset = load_dataset("json", data_files="/home/jovyan/visual-thinker-workspace/synthetic_data/verifiers/tool_categorized.jsonl", split='train')
dataset = dataset.map(lambda x: {"new_answers": x['generations'] if x.get('generations') and x['generations'] else ''})
dataset = dataset.remove_columns(['generations', 'finish_reasons', 'api_metadata'])
print(dataset)
print(dataset[0])

def count_correct_new_answers(example):
    original_answer = example.get('answer')
    new_answers_list = example.get('new_answers')
    count = 0
    if not isinstance(original_answer, str) or not isinstance(new_answers_list, list):
        return {'correct_new_answers_count': 0}
    processed_original_answer = original_answer.strip().lower()
    if not processed_original_answer:
        return {'correct_new_answers_count': 0}
    for new_answer in new_answers_list:
        if isinstance(new_answer, str):
            processed_new_answer = new_answer.strip().lower()
            if processed_original_answer in processed_new_answer:
                count += 1
    return {'correct_new_answers_count': count}

updated_dataset = dataset.map(count_correct_new_answers)
print("\nUpdated Dataset:")
print(updated_dataset)
print("\nExample 0 with count:")
print(updated_dataset[0])
print("\nExample 1 with count:")
print(updated_dataset[1])

total_new_answers = sum(len(na_list) if isinstance(na_list, list) else 0 for na_list in updated_dataset['new_answers'])
total_correct_new_answers = sum(updated_dataset['correct_new_answers_count'])
if total_new_answers > 0:
    overall_accuracy = (total_correct_new_answers / total_new_answers) * 100
    print(f"\nTotal New Answers checked: {total_new_answers}")
    print(f"Total Correct New Answers found: {total_correct_new_answers}")
    print(f"Overall Accuracy (based on substring check): {overall_accuracy:.2f}%")
else:
    print("\nNo new answers found to calculate overall accuracy.")

print("\nFiltering into separate datasets...")
dataset_0_correct = updated_dataset.filter(lambda example: example['correct_new_answers_count'] == 0)
dataset_1_correct = updated_dataset.filter(lambda example: example['correct_new_answers_count'] == 1)
dataset_2_correct = updated_dataset.filter(lambda example: example['correct_new_answers_count'] == 2)
dataset_3plus_correct = updated_dataset.filter(lambda example: example['correct_new_answers_count'] >= 3)
print("Filtering complete.")

total_rows = (len(dataset_0_correct) + len(dataset_1_correct) + len(dataset_2_correct) + len(dataset_3plus_correct))
print(f"\nTotal rows in original dataset: {len(updated_dataset)}")
print(f"Total rows across filtered datasets: {total_rows}")
assert len(updated_dataset) == total_rows, "Mismatch in row counts after filtering!"
print("Row counts match.")

dataset_0_correct.push_to_hub("HoangHa/Tool-RL","0correct", token = "")
dataset_1_correct.push_to_hub("HoangHa/Tool-RL","1correct", token = "")
dataset_2_correct.push_to_hub("HoangHa/Tool-RL","2correct", token = "")
dataset_3plus_correct.push_to_hub("HoangHa/Tool-RL","3correct", token = "")

print("--- Initial Dataset Sizes ---")
initial_len_0 = len(dataset_0_correct)
initial_len_3plus = len(dataset_3plus_correct)
print(f"0 Correct: {initial_len_0}")
print(f"3+ Correct: {initial_len_3plus}")

target_eval_samples_per_cat = 50
eval_samples_per_cat = min(target_eval_samples_per_cat, initial_len_0, initial_len_3plus)
print(f"\nTarget evaluation samples per category: {target_eval_samples_per_cat}")
if eval_samples_per_cat < target_eval_samples_per_cat:
    print(f"Warning: Not enough data for target eval size. Using {eval_samples_per_cat} samples per category instead.")
else:
    print(f"Will select {eval_samples_per_cat} samples per category for the evaluation set.")

seed = 42
shuffled_0 = dataset_0_correct.shuffle(seed=seed) if initial_len_0 > 0 else dataset_0_correct
shuffled_3plus = dataset_3plus_correct.shuffle(seed=seed) if initial_len_3plus > 0 else dataset_3plus_correct

selected_eval_sets = []
evaluation_dataset_temp = None

if eval_samples_per_cat > 0:
    eval_0 = shuffled_0.select(range(eval_samples_per_cat))
    eval_0_with_research_str = eval_0.add_column(name="research_str", column=['Yes'] * eval_samples_per_cat)
    selected_eval_sets.append(eval_0_with_research_str)
    print(f"Selected {len(eval_0_with_research_str)} samples from 0-correct for evaluation.")
    eval_3plus = shuffled_3plus.select(range(eval_samples_per_cat))
    eval_3plus_with_research_str = eval_3plus.add_column(name="research_str", column=['No'] * eval_samples_per_cat)
    selected_eval_sets.append(eval_3plus_with_research_str)
    print(f"Selected {len(eval_3plus_with_research_str)} samples from 3+ correct for evaluation.")
    evaluation_dataset_temp = concatenate_datasets(selected_eval_sets).shuffle(seed=seed)
else:
    print("\nCannot create an evaluation set (required size is 0 or input datasets are too small).")

remaining_indices_0 = range(eval_samples_per_cat, initial_len_0)
remaining_indices_3plus = range(eval_samples_per_cat, initial_len_3plus)
remaining_0 = shuffled_0.select(remaining_indices_0) if len(remaining_indices_0) > 0 else Dataset.from_dict({k: [] for k in shuffled_0.features})
remaining_3plus = shuffled_3plus.select(remaining_indices_3plus) if len(remaining_indices_3plus) > 0 else Dataset.from_dict({k: [] for k in shuffled_3plus.features})
print(f"\nRemaining samples for training: {len(remaining_0)} (0-correct), {len(remaining_3plus)} (3+ correct)")

train_samples_per_cat = min(len(remaining_0), len(remaining_3plus))
print(f"Will select {train_samples_per_cat} samples per category for the balanced training set.")

selected_train_sets = []
research_dataset_temp = None

if train_samples_per_cat > 0:
    train_0 = remaining_0.select(range(train_samples_per_cat))
    train_0_with_research_str = train_0.add_column(name="research_str", column=['Yes'] * train_samples_per_cat)
    selected_train_sets.append(train_0_with_research_str)
    print(f"Selected {len(train_0_with_research_str)} samples from remaining 0-correct for training.")
    train_3plus = remaining_3plus.select(range(train_samples_per_cat))
    train_3plus_with_research_str = train_3plus.add_column(name="research_str", column=['No'] * train_samples_per_cat)
    selected_train_sets.append(train_3plus_with_research_str)
    print(f"Selected {len(train_3plus_with_research_str)} samples from remaining 3+ correct for training.")
    research_dataset_temp = concatenate_datasets(selected_train_sets).shuffle(seed=seed)
else:
    print("\nCannot create a balanced training set from the remaining data.")

def convert_to_bool(example):
    return {'research': True if example['research_str'] == 'Yes' else False}

evaluation_dataset = None
research_dataset = None

if evaluation_dataset_temp:
    print("\nConverting 'research' column to boolean for evaluation dataset...")
    evaluation_dataset = evaluation_dataset_temp.map(convert_to_bool, remove_columns=['research_str'])
    print("Conversion complete.")
    print("Evaluation dataset features:", evaluation_dataset.features)

if research_dataset_temp:
    print("\nConverting 'research' column to boolean for training dataset...")
    research_dataset = research_dataset_temp.map(convert_to_bool, remove_columns=['research_str'])
    print("Conversion complete.")
    print("Training dataset features:", research_dataset.features)

print("\n--- Summary ---")
if evaluation_dataset:
    print(f"Evaluation Dataset Size: {len(evaluation_dataset)}")
    eval_bool_values = np.array(evaluation_dataset['research'])
    true_count_eval = np.sum(eval_bool_values)
    false_count_eval = len(eval_bool_values) - true_count_eval
    print(f"  Eval 'research' counts: True={true_count_eval}, False={false_count_eval}")
else:
    print("Evaluation Dataset: Not created.")

if research_dataset:
    print(f"Training Dataset (research_dataset) Size: {len(research_dataset)}")
    train_bool_values = np.array(research_dataset['research'])
    true_count_train = np.sum(train_bool_values)
    false_count_train = len(train_bool_values) - true_count_train
    print(f"  Train 'research' counts: True={true_count_train}, False={false_count_train}")
else:
    print("Training Dataset (research_dataset): Not created.")

if evaluation_dataset and research_dataset:
    total_selected_0 = len(evaluation_dataset.filter(lambda x: x['research']==True)) + len(research_dataset.filter(lambda x: x['research']==True))
    total_selected_3plus = len(evaluation_dataset.filter(lambda x: x['research']==False)) + len(research_dataset.filter(lambda x: x['research']==False))
    print(f"\nTotal selected corresponding to 0-correct (True): {total_selected_0}")
    print(f"Total selected corresponding to 3+ correct (False): {total_selected_3plus}")

research_dataset = research_dataset.remove_columns(['new_answers', 'correct_new_answers_count'])
evaluation_dataset = evaluation_dataset.remove_columns(['new_answers', 'correct_new_answers_count'])

research_dataset.push_to_hub("HoangHa/Tool-RL","train", token = "")
evaluation_dataset.push_to_hub("HoangHa/Tool-RL","test", token = "")