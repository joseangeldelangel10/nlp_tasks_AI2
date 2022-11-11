# NLP Module Project - ITESM CEM 
***By: Jose Angel Del Angel Dominguez  A01749386***

## Description

This repo contains my solution of the NLP module project for the advanced AI course, 
on the root dir there is a run.py file that will prompt the user for a Hugging Face API token,
after a valid token is provided the file will print on stdout the solution for each task.

The tasks required are:
1. Evaluate the sentiment of a set of movie reviews (POSITIVE, NEGATIVE) using a pretrained model
2. Retrain a NER model using a dataset with tagged tweets
3. Evaluate the perfomance of two translation APIs using BLEU score

To generate a huggingface token sign-in into huggingface, go to https://huggingface.co/settings/tokens
and generate a token with permission to read models

### Installing and excecuting code 
1. Clone the repository in your local machine
2. Open the command line interface (CLI) 
3. install the required libraries using ```pip install -r requirements.txt```
4. Ensure that you have a huggingface token with permission to read models
5. Get a DeepL translator API key at https://www.deepl.com/en/docs-api
6. Once you have the DeepL API key, create a file called ```api_keys.py``` under the dir ```./task_3/```. The file should contain the following:
```
DEEPL_KEY = <your_deepl_api_key_as_str>
# (for gracemikaela@gmail.com a DeepL API key should be available as a canvas comment or file)
```
7. run the main script using  ```python run.py```


Once you run the script the output generated should be similar to the following:
* for task 1:

![image](task_1_output.png)

* for task 2:

![image](task_2_output.png)

* for task 3:

![image](task_3_output.png)

### Tests: 

To run the unit tests for the porject methods log in to huggingface cli using your huggingface token and then from the root dir of the repo, call: 
```
python -m pytest tests.py
```
You should get an output as the following:

![tests_output](tests_output.png)

### Additional notes on Task 2: 

The training loss / validation loss graph when we use the labels found within the whole twitter dataset is the following:

![full_training_graph](task_2_full_training.png)

On the Y axis we have the value of the training and validation losses for the full dataset and on the X axis we have the epoch number.

We can see that when using these new labels, the model quickly overfits since validation loss gets stucked on a high value while the training loss decreases. We can assume this is caused by the low volume of data and the huge number of trainable parameters we have.

To solve this we adopted another aproach, which would be useful if we wanted to retrain the model to improve on detecting the same NERs but on tweets.

On this aproach we remap the labels found on the twitter dataset the following way: 
* "company"->"ORG",
* "facility"->"O",
* "geo-loc"->"LOC",
* "movie"->"O",
* "musicartist"->"O",
* "other"->"O",
* "person"->"PER",
* "product"->"O",
* "sportsteam"->"ORG",
* "tvshow"->"O"
If we retrain the model after doing this we have the following results:

![full_training_graph](task_2_full_training_remaped.png)

As we can se the model was mostly ready to be used on tweets, since it had low values from the initial training and validation losses. 

However, on the retraining, the validation loss reduced a little bit during the first epochs, which hopefully led us to a model that is a somewhat better when detecting our NERs on tweets.

## Authors

Contributors names and contact info:

Jose Angel Del Angel Dominguez  
[joseangeldelangel@gmail.com](mailto:joseangeldelangel@gmail.com)
[website](joseangeldelangel.com)
[linkedin](https://www.linkedin.com/in/jos%C3%A9-%C3%A1ngel-del-%C3%A1ngel-6a9293175/)