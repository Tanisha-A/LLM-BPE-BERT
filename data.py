from datasets import load_dataset

dataset = load_dataset("Skylion007/openwebtext")

# Save first 100 samples (or more) to a text file
with open("openwebtext_sample.txt", "w") as f:
    for example in dataset['train'].select(range(100)):
        f.write(example['text'] + "\n\n")