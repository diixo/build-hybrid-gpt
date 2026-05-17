from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from tqdm import tqdm
import json
from datasets import Dataset, concatenate_datasets


TRAIN = False

EOT = "<|endoftext|>"

tokenizer_path = "data/noomo-32k"

def read_jsonl(file_path: str) -> list:
    text = []
    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if item.get("example"):
                text.append(item["example"])
            else:
                title = item["title"]
                description = item["description"]
                text.append(title + " " + description)
    return text


if TRAIN:

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    fw_extended = concatenate_datasets([
        #Dataset.from_dict({ "text": read_jsonl("data/dictionary.cambridge.org-00.jsonl") }),
        #Dataset.from_dict({ "text": read_jsonl("data/dictionary.cambridge.org-01.jsonl") }),
        fw,
        Dataset.from_dict({ "text": read_jsonl("datasets/arxiv-corpus/arxiv_cs_2015_2020.jsonl") }),
        Dataset.from_dict({ "text": read_jsonl("datasets/arxiv-corpus/arxiv_cs_2021_2024.jsonl") }),

        ])

    total_rows = len(fw_extended)

    print(f"total_rows: {total_rows}")  # 11104126

    ################################################################################################

    def text_iterator():
        for row in tqdm(fw_extended, total=total_rows, unit="rows"):
            txt = row.get("text", "")
            if txt and isinstance(txt, str):
                yield txt

    ################################################################################################

    raw_tokenizer = Tokenizer(BPE())
    raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    raw_tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=32_256,
        min_frequency=5,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[]
    )

    raw_tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    raw_tokenizer.add_special_tokens([EOT])


    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = raw_tokenizer,
        eos_token = EOT,
        bos_token = None,
        unk_token = None
    )

    fast_tokenizer.save_pretrained(tokenizer_path)

#############################################################################################################

# export to gpt2 format for compatibility

tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True, add_prefix_space=True)


added = tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "bos_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|knowledge|>",
        "<|instruction|>",
        "###",
        ]
})

print(f"added: {added}, vocab_size: {len(tokenizer)}")

tokenizer.save_pretrained("data/gpt-noomo-32k")

test_text = "<|user|> What is the capital of France? <|assistant|> Paris. <|endoftext|>"

test_text = "<|user|> GPT is a type of large language model. <|assistant|> The chatGPT and other GPTs are based on a deep learning architecture called the transformer. <|endoftext|>"

print(f"Tokens: {tokenizer.tokenize(test_text)}")

print(f"Tokens: {tokenizer.tokenize('###What is Wikipedia? Assistant: ### Wikipedia is a free online encyclopedia.')}")
