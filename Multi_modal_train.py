

import torch
import torchaudio
from transformers import Wav2Vec2Model, HubertModel, HubertForSequenceClassification, Wav2Vec2Processor,PretrainedConfig
import wandb
wandb.init()

# Load the HuBERT model and its tokenizer
NUM_LABELS = 5
model_path = "facebook/hubert-large-ll60k"
config = PretrainedConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
config.feat_proj_layer_norm = True
config.use_weighted_layer_sum = True
config.classifier_proj_size = 1024
config.mask_time_min_masks = 2
config.mask_time_prob = 0
hubert_model = HubertForSequenceClassification.from_pretrained(model_path,config=config,ignore_mismatched_sizes=True )


from datasets import load_from_disk

# Load the dataset dict from a file
ds = load_from_disk('/home/adham.ibrahim/Emo_Rec/multi_modal/dataset_speech_text_Embeddings')

# convert the Hugging Face Dataset object back to a Pandas DataFrame
df = ds.to_pandas()

import torch.utils.data as data
from datasets import Dataset ,DatasetDict, load_metric
# INTRODUCE TRAIN TEST VAL SPLITS

# 90% train, 10% test + validation
train_testvalid = ds.train_test_split(shuffle=True, test_size=0.1)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})


import transformers

from transformers import TrainingArguments


trainer_config = {
  "OUTPUT_DIR": "results",
  "TRAIN_EPOCHS": 40,
  "TRAIN_BATCH_SIZE": 30,
  "EVAL_BATCH_SIZE": 30,
  "GRADIENT_ACCUMULATION_STEPS": 2,
  "WARMUP_STEPS": 300,
  "DECAY": 0.01,
  "LOGGING_STEPS": 10,
  "MODEL_DIR": "models/test-hubert-model",
  "SAVE_STEPS": 100,
  "LR": 0.0000001,
}

# Fine-Tuning with Trainer
training_args = TrainingArguments(
    output_dir=trainer_config["OUTPUT_DIR"],  # output directory
    gradient_accumulation_steps=trainer_config[
        "GRADIENT_ACCUMULATION_STEPS"
    ],  # accumulate the gradients before running optimization step
    num_train_epochs=trainer_config[
        "TRAIN_EPOCHS"
    ],  # total number of training epochs
    per_device_train_batch_size=trainer_config[
        "TRAIN_BATCH_SIZE"
    ],  # batch size per device during training
    per_device_eval_batch_size=trainer_config[
        "EVAL_BATCH_SIZE"
    ],  # batch size for evaluation
    warmup_steps=trainer_config[
        "WARMUP_STEPS"
    ],  # number of warmup steps for learning rate scheduler
    save_steps=trainer_config["SAVE_STEPS"], # save checkpoint every 100 steps
    weight_decay=trainer_config["DECAY"],  # strength of weight decay
    logging_steps=trainer_config["LOGGING_STEPS"],
    evaluation_strategy="epoch",  # report metric at end of each epoch
    report_to="wandb",  # enable logging to W&B
    learning_rate=trainer_config["LR"]
)
#Fine-tune the HuBERT model on the emotion recognition task using the new dataset
#from dataclasses import asdict

#train_dataset[0]
trainer = transformers.Trainer(
    model=hubert_model,
    train_dataset=ds["train"],  # training dataset
    eval_dataset=ds["val"],  # evaluation dataset
    args=training_args,
)
trainer.train()


trainer.save_model("multimodal__trained_model_V3_30epochs")

print("model saved")
# TO RESUME TRAINING FROM CHECKPOINT
# trainer.train("results/checkpoint-2000")

# import logging

# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
# )
# # VALIDATION SET RESULTS
# logging.info("Eval Set Result: {}".format(trainer.evaluate()))

# print("Eval Set Result: {}".format(trainer.evaluate()))
# # TEST RESULTS
# test_results = trainer.predict(ds["test"])
# print("Test Set Result: {}".format(test_results.metrics))
# #logging.info("Test Set Result: {}".format(test_results.metrics))
# #wandb.log({"test_accuracy": test_results.metrics["test_accuracy"]})
# print("Test Set Result: {}".format(test_results.metrics["test_accuracy"]))
# import os

# PROJECT_ROOT = "/home/adham.ibrahim/Emo_Rec/multi_modal"

# trainer.save_model(os.path.join(PROJECT_ROOT, trainer_config["MODEL_DIR"]))

# # logging trained models to wandb
# wandb.save(
#     os.path.join(PROJECT_ROOT, trainer_config["MODEL_DIR"], "*"),
#     base_path=os.path.dirname(trainer_config["MODEL_DIR"]),
#     policy="end",
# )
