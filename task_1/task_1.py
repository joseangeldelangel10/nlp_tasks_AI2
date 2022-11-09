# the model that we will use for task 1 (sentiment analysis on movie reviews)
# requires the hugging face transformers API that is installed using:
#   pip install transformers    
# We also need to have the huggingface_hub installed using:
#   pip install huggingface_hub
# We also need to have pytorch installed in my own case for windows to run on cpu I can use:
#   pip3 install torch torchvision torchaudio
# we also need to be signed up on hugging face and to have an authentication token
'''
TODO: add additional documentation of the file
Model link: https://huggingface.co/JamesH/Movie_review_sentiment_analysis_model?text=I+love+AutoTrain+%F0%9F%A4%97
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
        # Model output values are given in the order [negative_review_probability, positive_review_probability]
        # Output class is SequenceClassifierOutput here we extract the probabilities print the prediction value
        # based on the biggest probability 
        outputs = model(**inputs)
        redable_outputs = outputs.to_tuple()
        redable_outputs = redable_outputs[0]
        negative_prob = redable_outputs[0][0].item()
        positive_prob = redable_outputs[0][1].item()
        print("POSITIVE" if (positive_prob>=negative_prob) else "NEGATIVE")

if __name__ == "__main__":
    task_1()