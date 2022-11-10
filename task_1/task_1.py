'''
This code takes a pretrained NLP model for sentiment analysis on movie reviews to  
predict the sentiment from each movie review found in tiny_movie_reviews_dataset.txt, 
the model used can be found on:

https://huggingface.co/JamesH/Movie_review_sentiment_analysis_model?text=I+love+AutoTrain+%F0%9F%A4%97
'''
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def task_1(calling_dir = "."):    
    # We load the pre-trained model for movie reviews
    model = AutoModelForSequenceClassification.from_pretrained("JamesH/autotrain-third-project-1883864250", use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained("JamesH/autotrain-third-project-1883864250", use_auth_token=True)
    # We open the movie reviews file and ask the model to predict the sentiment of each movie review 
    movie_reviews_file = open( os.path.join(calling_dir, "tiny_movie_reviews_dataset.txt") , "r")
    for review in movie_reviews_file:
        inputs = tokenizer(review, return_tensors="pt")
        # Model output values are given in the format ([[[negative_review_probability, positive_review_probability]]])
        # here we extract the probabilities of each sentiment and print "POSSITIVE" or "NEGATIVE"  
        # based on the biggest probability 
        outputs = model(**inputs)
        redable_outputs = outputs.to_tuple()
        redable_outputs = redable_outputs[0]
        negative_prob = redable_outputs[0][0].item()
        positive_prob = redable_outputs[0][1].item()
        print("POSITIVE" if (positive_prob>=negative_prob) else "NEGATIVE")

if __name__ == "__main__":
    task_1()