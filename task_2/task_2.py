'''
This code takes a pretrained NLP model for named entity recognition (NER), such model has been trained to
recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC)
and will be retrained to recognize 10 types of entities 'geo-loc' 'facility' 'movie' 'company' 'product'
'person' 'sportsteam' 'other' 'tvshow' 'musicartist', using a dataset with Tweets and its respective
tags within the tweets ('twitter_dataset_dev.xlxs'). To see more about the dataset, the number of labels
and other details you can check the data_analysis notebook on this same directory.

The model used can be found on:

https://huggingface.co/dslim/bert-base-NER?text=My+name+is+Sarah+and+I+live+in+London
'''

from datasets import Dataset, Features, Value, ClassLabel, Sequence
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainerCallback, TrainingArguments, BertConfig, DataCollatorForTokenClassification
import pandas as pd
import numpy as np
import evaluate
import copy
import matplotlib.pyplot as plt
import os

N_EXAMPLES_TO_TRAIN = 50
N_TRAINING_EPOCHS = 20
TRAINING_LEARNING_RATE = 2e-5

class NERDataMaker:
    """
    class that is initialized with a list containing lists of tuples, 
    each inner list contains tuples with word, tag pairs. 
    
    Every list on the entry list corresponds to a tweet on the dataset from 
    'aritter' (ritter.1492@osu.edu), found on:

    https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16

    This class helps us to vectorize the labels and words of the given dataset, 
    the constructor method (__init__) vectorizes labels and leaves the id2label and label2id
    properties ready. 
    
    The as_hf_dataset method vectorizes the words on the given dataset and,
    changes the dtype of the dataset to a transformers.Dataset (huggingface native), while 
    adding the necessary items to the new huggingface dataset.

    The rest of the methods are helper methods.

    This class is based on the original class of Sanjaya Subedi, referenced on his
    tutorial at: 
    https://sanjayasubedi.com.np/deeplearning/training-ner-with-huggingface-transformer/
    """
    def __init__(self, tokens_with_entities_list):
        self.unique_entities = []
        self.processed_texts = []

        temp_processed_texts = []
        for tokens_with_entities in tokens_with_entities_list:            
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)

        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")

        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities]) # <- after this line classifier labels are converted into a numeric value

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)

            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }

        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def string_starts_with(self, string, candidate_starting_substring):
      len_candidate = len(candidate_starting_substring)      
      return string[:len_candidate] == candidate_starting_substring

    def remove_starting_hastags(self, string):
      new_string = string
      for char in list(new_string):
        if char != "#":
          break
        else:
          new_string = new_string[1:]
      return new_string    

    def as_hf_dataset(self, tokenizer):
        def tokenize_and_align_labels(examples):
            '''
            This funtion takes a list of word-NER_tag pairs, tokenizes the words and assigns the label of the
            word with the starting token (substring) of the word, the rest of the tokens (substrings) within the
            word will have a value of -100 assigned, which refers to a that a huggingface model should ignore.
            
            See the following example for more details.

            example input: [( '@sammuelLJackson', PER)]
            example output: [ ('@',PER), ('sam', -100), ('uel', -100), (L, -100), ('Jackson', -100) ]

            '''
            original_words_list = examples["tokens"]
            labels_for_original_words_list = examples["ner_tags"]
            tokenized_inputs = tokenizer(original_words_list, truncation=True, is_split_into_words=True, padding = True)
            subwords_after_tokenizing = tokenizer.convert_ids_to_tokens(ids = tokenized_inputs["input_ids"])

            labels = []
            word_index = 0
            rebuilt_string = ""
            added_starting_label = False            
            for subword in subwords_after_tokenizing:            
              #special cases:              
              if subword is None or subword == "[CLS]" or subword == "[SEP]":
                labels.append(-100)
              else:
                if word_index < len(original_words_list):
                  rebuilt_string += self.remove_starting_hastags(subword)
                  if self.string_starts_with(original_words_list[word_index], subword) and not added_starting_label:
                    labels.append(labels_for_original_words_list[word_index])
                    added_starting_label = True
                  else:
                    labels.append(-100)
                  if rebuilt_string == self.remove_starting_hastags(original_words_list[word_index]):
                    word_index += 1
                    rebuilt_string = ""
                    added_starting_label = False                              

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        ids, ner_tags, tokens = [], [], []
        print("processed texts:")
        print(self.processed_texts[0])
        print("unique entities:")
        print(self.unique_entities)
        for i, pt in enumerate(self.processed_texts):
            ids.append(i)
            pt_tokens,pt_tags = list(zip(*pt))
            ner_tags.append(pt_tags)
            tokens.append(pt_tokens)
        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens
        }
        features = Features({
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities)),
            "id": Value("int32")
        })
        ds = Dataset.from_dict(data, features)        
        tokenized_ds = ds.map(tokenize_and_align_labels, batched=False) # TODO: change this to true latter to achieve a batched behaiviour
        return tokenized_ds

def create_tweets_with_entities_list(train_df):
    '''
    This function takes a dataframe with columns 'WORD' and 'LABEL' 
    (the shape of our dataset) where every row containing NA represents
    the end of a tweet
    '''
    is_a_space = pd.isnull(train_df["WORD"])
    annotated_tweets = []
    annotated_tweet = []
    for indx in train_df.index:
        if is_a_space[indx]:
            if len(annotated_tweet) > 0:
                annotated_tweets.append( copy.deepcopy(annotated_tweet) )
            annotated_tweet = []
        else:
            annotated_tweet.append((train_df["WORD"][indx], train_df["LABEL"][indx])) 
    if len(annotated_tweet) > 0:
        annotated_tweets.append( copy.deepcopy(annotated_tweet) )
    return annotated_tweets

def task_2(calling_dir = "."):
    '''
    Main function for the task
    '''    
    global N_EXAMPLES_TO_TRAIN
    global N_TRAINING_EPOCHS
    global TRAINING_LEARNING_RATE
    # we read the dataset
    data_df = pd.read_excel( os.path.join(calling_dir, "twitter_dataset_train.xlsx") )

    # we transform the each tweet to a list of word-label pairs
    tweets_with_entities = create_tweets_with_entities_list(data_df)

    # we make the train validation split
    num_entries = len(tweets_with_entities)
    tweets_with_entities = tweets_with_entities[:N_EXAMPLES_TO_TRAIN%num_entries]        
    train_tweets_with_entities = tweets_with_entities[:int(N_EXAMPLES_TO_TRAIN*0.8)]
    eval_tweets_with_entities = tweets_with_entities[int(N_EXAMPLES_TO_TRAIN*0.8):]

    # we transform the datset to a huggingface Dataset  
    training_data = NERDataMaker( train_tweets_with_entities )
    eval_data = NERDataMaker( eval_tweets_with_entities )
    tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_data_prepared_for_hf_training = training_data.as_hf_dataset(tokenizer=tokenizer)
    eval_data_prepared_for_hf_training = eval_data.as_hf_dataset(tokenizer=tokenizer)

    # we import the pretrained model
    model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER",
                                                   ignore_mismatched_sizes=True,                                                   
                                                   _num_labels=len(training_data.unique_entities),                                                    
                                                   id2label=training_data.id2label, 
                                                   label2id=training_data.label2id)

    # we declare the training params
    training_arguments = TrainingArguments(output_dir = os.path.join( calling_dir, "NER_models") ,
                                        evaluation_strategy="epoch",
                                        logging_strategy="epoch",
                                        learning_rate=TRAINING_LEARNING_RATE,
                                        per_device_train_batch_size=16,
                                        per_device_eval_batch_size=16,                                      
                                        num_train_epochs=N_TRAINING_EPOCHS,
                                        weight_decay=0.01)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)    
        predictions = predictions.astype('int32')
        labels = labels.astype('int32')
        predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1],))
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],))          
        return metric.compute(predictions=predictions, references=labels)

    # we retrain the model
    trainer = Trainer(model = model,
                  args = training_arguments,
                  train_dataset=train_data_prepared_for_hf_training,
                  eval_dataset=eval_data_prepared_for_hf_training,
                  data_collator=data_collator,
                  compute_metrics = compute_metrics)

    trainer.train()

    # we format the training results for plotting
    raw_history = trainer.state.log_history[:-1]
    training_metrics = {"eval_loss":[], "eval_accuracy": [], "loss":[]}
    current_epoch = 1.0
    for dictionary in raw_history:
        if dictionary["epoch"] != current_epoch:    
            current_epoch += 1.0
        metrics_in_dict = [i for i in training_metrics.keys() if i in dictionary.keys()]  
        for m in metrics_in_dict:
            training_metrics[m].append( (int(current_epoch),dictionary[m]) )

    # we plot the training results
    plt.plot(list(zip(*training_metrics["loss"]))[1], label="loss")
    plt.plot(list(zip(*training_metrics["eval_loss"]))[1], label="val_loss")
    plt.legend(loc='best')
    plt.show()    

if __name__ == "__main__":    
    task_2()