import task_1.task_1 as t1
import task_2.task_2 as t2
import task_3.task_3 as t3
from task_2.task_2 import NERDataMaker
import pandas as pd
from task_3.api_keys import DEEPL_KEY
from datasets import Dataset
from transformers import BertTokenizer
import os

def test_task_1_classify_movie_review_sentiment():
    t1.load_hugging_face_model("JamesH/autotrain-third-project-1883864250")
    test_movie_review_1 = "once again the director delivered an awfull pice, completly messy and absolutetly nonsense"
    test_movie_review_2 = "one of the best movies I've ever seen, a delightfull film"
    sentiment_on_review_1 = t1.classify_movie_review_sentiment(test_movie_review_1)
    sentiment_on_review_2 = t1.classify_movie_review_sentiment(test_movie_review_2)
    assert sentiment_on_review_1 == "NEGATIVE"
    assert sentiment_on_review_2 == "POSITIVE"

def test_task_1_main_function():
    current_dir = os.getcwd()
    task_1_script_dir = os.path.join(current_dir, "task_1")
    task_1_results = t1.task_1(calling_dir=task_1_script_dir) # testing that no exceptions occur when running test 1
    assert task_1_results[0] > 0, "error in test no negative reviws where found, please check model performance"
    assert task_1_results[1] > 0, "error in test no positive reviws where found, please check model performance"

def test_task_2_create_tweets_with_entities():
    # checking that the function generates lists with word tag pairs given a dataframe    
    dummy_twitter_dataset_dict = {
        "WORD":["@myself", "I", "am", "really", "sad", "that", "the", "imagine", "dragons", "concert", "at", "mexico", "city", "was", "canceled"], 
        "LABEL":[ "B-Person", "B-Person", "O", "O", "O", "O", "O", "B-musicartist", "I-musicartist", "O", "O", "B-geo-loc", "I-geo-loc", "O", "O" ]
        }
    expected_result = [[('@myself', 'B-Person'),
        ('I', 'B-Person'),
        ('am', 'O'),
        ('really', 'O'),
        ('sad', 'O'),
        ('that', 'O'),
        ('the', 'O'),
        ('imagine', 'B-musicartist'),
        ('dragons', 'I-musicartist'),
        ('concert', 'O'),
        ('at', 'O'),
        ('mexico', 'B-geo-loc'),
        ('city', 'I-geo-loc'),
        ('was', 'O'),
        ('canceled', 'O')]]
    dummy_twitter_data_df = pd.DataFrame.from_dict(dummy_twitter_dataset_dict)
    processed_dataframe_result = t2.create_tweets_with_entities_list(dummy_twitter_data_df)    
    assert processed_dataframe_result == expected_result

def test_task_2_NERDataMaker():
    # ============ TESTING CLASS CONSTRUCTOR ============

    test_dataset = [[('@myself', 'B-Person'),
        ('I', 'B-Person'),
        ('am', 'O'),
        ('really', 'O'),
        ('sad', 'O'),
        ('that', 'O'),
        ('the', 'O'),
        ('imagine', 'B-musicartist'),
        ('dragons', 'I-musicartist'),
        ('concert', 'O'),
        ('at', 'O'),
        ('mexico', 'B-geo-loc'),
        ('city', 'I-geo-loc'),
        ('was', 'O'),
        ('canceled', 'O')]]
    expected_unique_entities = ['O', 'B-Person', 'B-geo-loc', 'B-musicartist', 'I-geo-loc', 'I-musicartist']        
    data_maker = NERDataMaker(test_dataset)        
    assert data_maker.unique_entities == expected_unique_entities
    assert type(data_maker.processed_texts[0][0][0]) == str
    assert type(data_maker.processed_texts[0][0][1]) == int    
    
    # ============ TESTING CLASS METHOD STR STARTS WITH ============

    starts_with_ressult_1 = data_maker.string_starts_with("Programming", "Progra")
    starts_with_ressult_2 = data_maker.string_starts_with("Programming", "Pro")
    starts_with_ressult_3 = data_maker.string_starts_with("Programming", "Hello")
    assert starts_with_ressult_1 == True
    assert starts_with_ressult_2 == True
    assert starts_with_ressult_3 == False

    # ============ TESTING CLASS METHOD REMOVING STARTING HASTAGS WITH ============

    remove_starting_hastag_result_1 = data_maker.remove_starting_hastags("sam")
    remove_starting_hastag_result_2 = data_maker.remove_starting_hastags("##muel")
    remove_starting_hastag_result_3 = data_maker.remove_starting_hastags("hello#world")
    assert remove_starting_hastag_result_1 == "sam"
    assert remove_starting_hastag_result_2 == "muel"
    assert remove_starting_hastag_result_3 == "hello#world"

    # ============ TESTING CLASS METHOD AS HF DATASET ============

    tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
    hf_dataset = data_maker.as_hf_dataset(tokenizer)
    assert type(hf_dataset) == Dataset

def test_task_3_translate_to_english_using_deepl():
    test_word = "ciudad de mexico"
    expected_translations = ["mexico city", "city of mexico"]
    deepl_result = t3.translate_to_english_using_deepl(test_word, DEEPL_KEY)
    argostranslate_result = t3.translate_to_english_using_argostranslate(test_word)
    deepl_result = deepl_result.lower()
    argostranslate_result = argostranslate_result.lower()    
    assert deepl_result in expected_translations
    assert argostranslate_result in expected_translations
    