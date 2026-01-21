#!/usr/bin/env python
# coding: utf-8

####### import Packages ################

from transformers import AutoTokenizer, AutoModel
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn

# Initialize tokenizer and model from Snowflake Arctic Embed
tokenizer = AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-m')
model = AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-m')

######## training Dataset ################

txtDataPath = "./data_lm.txt"

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=txtDataPath,
    block_size=512,
)

##########  Training Arguments ##############

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./snowflake-arctic-embed-retrained",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    seed=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

###### Trainer function to train the model ###### 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

###### Save the trained model #########

trainer.save_model("./snowflake-arctic-embed-retrained")