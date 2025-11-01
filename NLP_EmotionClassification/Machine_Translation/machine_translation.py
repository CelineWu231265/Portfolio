# %% [markdown]
# ## Task 6 Machine Translation  
# 
# ### This was following the tutorial provided to us

# %%
import os
import sys
import transformers
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from ftfy import fix_text

# %%
model_checkpoint = "Helsinki-NLP/opus-mt-es-en" #fetching pretrained translation model from hugging face

# %%
raw_datasets = load_dataset("Helsinki-NLP/opus_books", "en-es") #getting english to spanish dataset

# %%
train_test_split = raw_datasets['train'].train_test_split(test_size=0.01)  # split 10% for test

validation_test_split = train_test_split['train'].train_test_split(test_size=0.01)  # 10% of 90% => 9% of original

# irganizing the splits into a new DatasetDict
raw_datasets = DatasetDict({
    'train': validation_test_split['train'],
    'validation': validation_test_split['test'],
    'test': train_test_split['test']
})

# %%
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
max_input_length = 128
max_target_length = 128 #max 128 tokens

source_lang = "es"
target_lang = "en" #setting it up so that it translates from spanish to english


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]] # it gathers spansish sentences as input and english sentences as targets 
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) # tokenizes the sentences

    # setting up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
preprocess_function(raw_datasets["train"][:2])

# %%
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True) #converts es to en translation pairs into tokenized tensors that the model can learn from 

# %%
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) # loads the appropriate architecure automatically 

# %%
batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1 # all parameters subject to change

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf") #formats and pads a list of examples (the data)

# %%
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

# %%
train_dataset = model.prepare_tf_dataset( #creates and prepares the dataset for training
    tokenized_datasets["test"], #it uses the test because the dataset is so large it would take far too long to train but it should be "train"
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset( #this one is user for validatioin in training
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

generation_dataset = model.prepare_tf_dataset( #uses validation set but for inference
    tokenized_datasets["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator,
)

# %%
optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)

# %%
model.fit(train_dataset, validation_data=validation_dataset, epochs=1) #training

# %%
model.save_pretrained("my_local_model")         # saves config + weights
tokenizer.save_pretrained("my_local_model")     # saves tokenizer files


# %%
model = TFAutoModelForSeq2SeqLM.from_pretrained("my_local_model")
tokenizer = AutoTokenizer.from_pretrained("my_local_model") #loading model and tokenizer

# %% [markdown]
# ### Code to translate spanish sentences from an excel file 

# %%
model_path = "my_local_model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

input_path = r"C:\Users\emilp\Documents\GitHub\2024-25c-fai2-adsai-EmilFox231007\datalab_tasks\Task_11\extracted_sentences.csv"    # <-- Replace with your Excel file path
df = pd.read_csv(input_path)

def translate(text):

    # Tokenize input text
    inputs = tokenizer.encode(text, return_tensors="tf", padding=True, truncation=True, max_length=256)
    # Generate translation
    outputs = model.generate(inputs, max_length=256)
    # Decode the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Apply translation to the column 
df['translated_sentence'] = df['Sentence'].apply(translate)

# save to excel file
output_path = r"C:\Users\emilp\Documents\GitHub\2024-25c-fai2-adsai-EmilFox231007\datalab_tasks\Task_11\extracted_sentences.csv"   # <-- Replace with desired output path
df.to_csv(output_path, index=False)



