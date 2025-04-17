from datasets import load_dataset, concatenate_datasets, Dataset
from collections import defaultdict, Counter
import random

squad_train = load_dataset("rajpurkar/squad", split="train")
squad_v2_train = load_dataset("rajpurkar/squad_v2", split="train")
squad_val = load_dataset("rajpurkar/squad", split="validation")
squad_v2_val = load_dataset("rajpurkar/squad_v2", split="validation")

full_dataset = concatenate_datasets([squad_train, squad_v2_train, squad_val, squad_v2_val])
print(len(set(full_dataset['title'])))

initial_len = len(full_dataset)
full_dataset = full_dataset.filter(lambda example: len(example['answers']['text']) > 0)
print(f"Filtered out {initial_len - len(full_dataset)} samples with no answers.")
print(f"Dataset size after filtering: {len(full_dataset)}")

multi_text = [i for i in range(len(full_dataset)) if len(full_dataset[i]['answers']['text']) > 1]
print(f"Number of samples with multiple answers: {len(multi_text)}")

def get_identical_answers_samples(dataset):
    identical = []
    for i in range(len(dataset)):
        answers = dataset[i]['answers']['text']
        if len(answers) > 1 and all(a == answers[0] for a in answers):
            identical.append(i)
    print(f"Number of samples with identical multiple answers: {len(identical)}")
    return identical

identical_indices = get_identical_answers_samples(full_dataset)
identical_dataset = Dataset.from_list([full_dataset[i] for i in identical_indices])
print(len(set(identical_dataset['title'])))

def preprocess_context_format(examples):
    contexts = []
    for title, context in zip(examples['title'], examples['context']):
        t = title.replace('_', ' ')
        contexts.append(f"# Title: {t}\n# Context: {context}")
    examples['context'] = contexts
    return examples

full_dataset = full_dataset.map(preprocess_context_format, batched=True, desc="Formatting context with title")
print("Sample after context preprocessing:")
print(full_dataset[0]['context'])

def preprocess_question_format(examples):
    questions = []
    for title, question in zip(examples['title'], examples['question']):
        t = title.replace('_', ' ')
        questions.append(f"# Title: {t}\n# Question: {question}")
    examples['question'] = questions
    return examples

full_dataset = full_dataset.map(preprocess_question_format, batched=True, desc="Formatting question with title")
print("Sample after question preprocessing:")
print(full_dataset[0]['question'])

def preprocess_answers(examples):
    new_answers = []
    for ans in examples['answers']:
        texts = ans['text']
        if not texts:
            new_answers.append("")
            continue
        if len(texts) > 1:
            count = Counter(texts)
            common, freq = count.most_common(1)[0]
            if freq > 1:
                new_answers.append(common)
                continue
            new_answers.append(min(texts, key=len))
        else:
            new_answers.append(texts[0])
    examples['answer'] = new_answers
    return examples

print("Preprocessing answers...")
full_dataset = full_dataset.map(preprocess_answers, batched=True, remove_columns=['answers'])
print("Sample after preprocessing:")
print(full_dataset[0])

print("Grouping data by title...")
title_to_indices = defaultdict(list)
for idx, title in enumerate(full_dataset['title']):
    title_to_indices[title].append(idx)
print(f"Found {len(title_to_indices)} unique titles with answers.")

train_indices = []
test_indices = []
SEED = 42
random.seed(SEED)

print("Splitting into train/test indices based on titles...")
for title, indices in title_to_indices.items():
    random.shuffle(indices)
    n = len(indices)
    if n >= 20:
        test_indices.append(indices[0])
        train_indices.extend(indices[1:20])
    elif n > 0:
        test_indices.append(indices[0])
        if n > 1:
            train_indices.extend(indices[1:])

print(f"Total train indices selected: {len(train_indices)}")
print(f"Total test indices selected: {len(test_indices)}")

train_indices.sort()
test_indices.sort()

train_dataset = full_dataset.select(train_indices)
test_dataset = full_dataset.select(test_indices)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

overlap = set(train_indices).intersection(set(test_indices))
print(f"Overlap between train and test indices: {len(overlap)}")

print("Creating corpus dataset...")
max_corpus_size = 10000
all_indices = set(range(len(full_dataset)))
remaining = list(all_indices - set(train_indices + test_indices))

num_additional = max(0, max_corpus_size - len(train_indices) - len(test_indices))
random.shuffle(remaining)
additional = remaining[:min(num_additional, len(remaining))]

corpus_indices = sorted(train_indices + test_indices + additional)
corpus_dataset = full_dataset.select(corpus_indices).shuffle(seed=SEED)

print(f"Final Corpus dataset size: {len(corpus_dataset)}")

print("\n--- Final Datasets ---")
print(f"Train Dataset ({len(train_dataset)} samples)")
print(f"Test Dataset ({len(test_dataset)} samples)")
print(f"Corpus Dataset ({len(corpus_dataset)} samples)")

train_counter = Counter(train_dataset['title'])
corpus_counter = Counter(corpus_dataset['title'])

print("Train set: Samples per title")
for title, count in train_counter.most_common():
    print(f"{title}: {count}")

print("\nCorpus set: Samples per title")
for title, count in corpus_counter.most_common():
    print(f"{title}: {count}")

print("\nSummary statistics:")
print(f"Unique titles in train: {len(train_counter)}")
print(f"Unique titles in corpus: {len(corpus_counter)}")
print(f"Titles with >1 samples in train: {sum(1 for c in train_counter.values() if c > 1)}")
print(f"Titles with >1 samples in corpus: {sum(1 for c in corpus_counter.values() if c > 1)}")
print(f"Titles with >5 samples in train: {sum(1 for c in train_counter.values() if c > 5)}")
print(f"Titles with >5 samples in corpus: {sum(1 for c in corpus_counter.values() if c > 5)}")

nikola_samples = train_dataset.filter(lambda x: x['title'] == 'University_of_Notre_Dame')
print(f"Found {len(nikola_samples)} samples with title 'Nikola_Tesla'")
for i, sample in enumerate(nikola_samples):
    print(f"\nSample {i+1}:\n{sample}")

train_dataset.push_to_hub("HoangHa/Tool-RL", "train-raw", token="")
# test_dataset.push_to_hub("HoangHa/Tool-RL", "test", token="") # At this phase, test set is not good -> move to the classification for better testset
corpus_dataset.push_to_hub("HoangHa/Tool-RL", "corpus", token="")