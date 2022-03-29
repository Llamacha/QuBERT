from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch
import json
import os
os.environ["WANDB_API_KEY"] = "3f053594ab7195d37a6f4279f73c0099603dbb93"

train_path = "../resource/train_normalized_bpeguided.txt"
eval_path = "../resource/test_normalized_bpeguided.txt"

# Initialize a tokenizer
tokenizer = Tokenizer(WordLevel())
tokenizer.pre_tokenizer = Whitespace()

# Customize training
trainer = WordLevelTrainer(vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.train([train_path], trainer)
tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

tokenizer.save("../quechuaBERT/tokenizer.json")

tokenizer.enable_truncation(max_length=512)

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer_config = {"max_len": 512}

with open("../quechuaBERT/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)

tokenizer = RobertaTokenizerFast.from_pretrained("../quechuaBERT", max_len=512)

model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128,
)

dataset_eval = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_path,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./BPEGuidedquechuaBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset_eval
)

trainer.train()

print(trainer.state.log_history[-2])

trainer.save_model("./quechuaBERT")