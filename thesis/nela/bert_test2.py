'''
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf
print(tf. __version__)


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = "Las abejas son verdes pero muchas veces tienen colores amarillos y anaranjados. Estas se alimentan de polen."
questions = ["De que color es la abeja?","De que se alimenta la abeja?","Que otros colores tienen?"]
 
def answer_question(question, text):
  # Function that simplifies answering a question
  for question in questions:
    # Concatenate the question and the textx
    inputs = tokenizer(question, text, add_special_tokens = True, return_tensors = 'tf')
    # Get the input ids (numbers) and convert to tokens (words)
    input_ids = inputs["input_ids"].numpy()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Run the pretrained model to get the logits (raw scores) for the scores
    output = qa_model(inputs)
    # Get the most likely beginning and end
    answer_start = tf.argmax(output.start_logits, axis = 1).numpy()[0]
    answer_end = (tf.argmax(output.end_logits, axis = 1)+1).numpy()[0]
    # Turn the tokens from the ids of the input string, indexed by the start and end tokens back into a string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print("Question {} \nAnswer: {}".format(question, answer))

answer_question(questions, text)
'''

#NEW TEST

def transform_labels(label):
    label = label['Sentiment']
    num = 0
    if label == 'Positive':
        num = 0
    elif label == 'Negative':
        num = 1
    elif label == 'Neutral':
        num = 2
    elif label == 'Extremely Positive':
        num = 3
    elif label == 'Extremely Negative':
        num = 4
    return {'labels': num}

def tokenize_data(example):
    return tokenizer(example['OriginalTweet'], padding='max_length')

from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': 'Corona_NLP_train.csv', 'test': 'Corona_NLP_test.csv'}, encoding = "ISO-8859-1")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

dataset = dataset.map(tokenize_data, batched=True)

remove_columns = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'Sentiment']
dataset = dataset.map(transform_labels, remove_columns=remove_columns)

from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer", num_train_epochs=3)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


train_dataset = dataset['train'].shuffle(seed=10).select(range(40000))
eval_dataset = dataset['train'].shuffle(seed=10).select(range(40000, 41000))

from transformers import Trainer

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
)

trainer.train()


import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer.evaluate()



# pip install transformers
# pip install tensorflow==2.3
# pip install datasets
# pip install torch


from datasets import load_dataset
imdb = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

