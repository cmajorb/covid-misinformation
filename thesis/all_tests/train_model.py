import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
import math


torch.cuda.empty_cache()
index = int(sys.argv[1])
#index = 0

hyper_parameters = {
  "with_training" : [True],
  "model_name" : ["dmis-lab/biobert-base-cased-v1.2"],
  "data_set" : ["/home/cmbrow38/covid-misinformation/thesis/coaid/train_clean.csv"],
  "per_gpu_batch_size": [4, 8],
  "learning_rate": [2e-5, 3e-5],
  "num_epochs": [4],
  "train_split": [0.05, 0.1]
}

configurations = []
for with_training in hyper_parameters["with_training"]:
    for model_name in hyper_parameters["model_name"]:
        for data_set in hyper_parameters["data_set"]:
            for per_gpu_batch_size in hyper_parameters["per_gpu_batch_size"]:
                for learning_rate in hyper_parameters["learning_rate"]:
                    for num_epochs in hyper_parameters["num_epochs"]:
                        for train_split in hyper_parameters["train_split"]:
                            configurations.append({
                                "with_training": with_training,
                                "model_name": model_name,
                                "data_set": data_set,
                                "per_gpu_batch_size": per_gpu_batch_size,
                                "learning_rate": learning_rate,
                                "num_epochs": num_epochs,
                                "train_split": train_split
                            })


config = configurations[index]
data_set_path = config["data_set"]
model_name = config["model_name"]
train_split = config["train_split"]
output_location = "/scratch/cmbrow38/coaid/final_test_"+model_name+"_"+str(index)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def transform_labels(label):
    label = label['output']
    return {'labels': label}

def tokenize_data(example):
    return tokenizer(example['input'], padding='max_length',max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = load_dataset('csv', data_files={'train': data_set_path}, encoding = "ISO-8859-1")
labels = list(set(dataset['train']['output']))
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
model = model.to(device)

dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.map(transform_labels)



training_args = TrainingArguments(output_location, 
                                num_train_epochs=config["num_epochs"],
                                per_device_train_batch_size=config["per_gpu_batch_size"],
                                evaluation_strategy="steps",
                                learning_rate=config["learning_rate"]
                                )
total_size = len(dataset['train'])
train_size = math.floor(total_size*train_split)
train_dataset = dataset['train'].shuffle(seed=10).select(range(train_size))
eval_dataset = dataset['train'].shuffle(seed=10).select(range(train_size, total_size))


metric = load_metric("accuracy")

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics
)

if(config["with_training"]):
    print("Running the training")
    trainer.train() #resume_from_checkpoint=True
else:
    print("Not training")

# results = trainer.evaluate()
# print(results)


#Get stats on dataset:
'''
import pandas as pd
df = pd.DataFrame(dataset['train'])
tokenized_articles_lengths=pd.DataFrame({'length': list(map(len, tokenizer(df['input'].to_list(), truncation=False, padding=False)['input_ids']))})
tokenized_articles_lengths.describe()

data_set_path = "/home/cmbrow38/covid-misinformation/thesis/nela/train_clean2.csv"
dataset = load_dataset('csv', data_files={'train': data_set_path}, encoding = "ISO-8859-1")

"/scratch/cmbrow38/coaid/final_test_"+model_name+"_"+str(sys.argv[1])
'''

from csv import writer

#Evaluate separate test
test_paths = ["test_data_nela.csv","test_data.csv"]
for test_path in test_paths:
    test_dataset = load_dataset('csv', data_files={'train': test_path}, encoding = "ISO-8859-1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = test_dataset.map(tokenize_data, batched=True)
    test_dataset = test_dataset.map(transform_labels)
    prediction = trainer.predict(test_dataset['train'])
    name = 'batch: ' + str(config['per_gpu_batch_size']) + ' lr: ' +  str(config['learning_rate']) + ' split: ' +  str(config['train_split'])
    with open('results.csv', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow([index, name, test_path, prediction.metrics["test_accuracy"]])  
        f_object.close()


import numpy as np
import pandas as pd
from torch import nn

test_dataset = load_dataset('csv', data_files="covid_tweets.csv", encoding = "ISO-8859-1")
#tokenizer = AutoTokenizer.from_pretrained(model_name)
test_dataset = test_dataset.map(tokenize_data, batched=True)
prediction = trainer.predict(test_dataset['train'])
preds = np.argmax(prediction.predictions, axis=-1)
confidence = nn.functional.softmax(torch.from_numpy(prediction.predictions), dim=-1)

df = pd.read_csv("covid_tweets.csv")
df['output'] = preds
df[['0','1']] = confidence

df.to_csv("output_logits.csv")


test_dataset = load_dataset('csv', data_files={'train': 'test_data.csv'}, encoding = "ISO-8859-1")
tokenizer = AutoTokenizer.from_pretrained(model_name)
test_dataset = test_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(transform_labels)
prediction = trainer.predict(test_dataset['train'])
prediction.predictions


