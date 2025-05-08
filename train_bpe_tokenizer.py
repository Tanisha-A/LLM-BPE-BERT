from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import os

# Input and output paths
input_file = "data/oasst1_finetune.txt"
output_dir = "bpe_tokenizer"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Train raw BPE tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=input_file,
    vocab_size=50257,
    min_frequency=2,
    special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ]
)

# Step 2: Save merges and vocab files (needed for compatibility)
tokenizer.save_model(output_dir)

# Step 3: Wrap using transformers' PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    vocab_file=os.path.join(output_dir, "vocab.json"),
    merges_file=os.path.join(output_dir, "merges.txt"),
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

# Step 4: Save in Hugging Face-compatible format
wrapped_tokenizer.save_pretrained(output_dir)

print(f"✅ BPE tokenizer saved to '{output_dir}' with tokenizer.json, vocab.json, merges.txt")