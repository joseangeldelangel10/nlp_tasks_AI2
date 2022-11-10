# NLP Module Project - ITESM CEM 
***By: Jose Angel Del Angel Dominguez***

## Description

This repo contains my solution of the NLP module project for the advanced AI course, 
on the root dir there is a run.py file that will prompt the user for a Hugging Face API token,
after a valid token is provided the file will print on stdout the solution for each task.

The tasks required are:
1 - Evaluate the sentiment of a set of movie reviews (POSITIVE, NEGATIVE) using a pretrained model
2 - Retrain a NER model using a dataset with tagged tweets
3 - Evaluate the perfomance of two translation APIs using BLEU score

To generate a huggingface token sign-in into huggingface, then go to https://huggingface.co/settings/tokens
and generate a token with permission to read models

The output generated should be similar to the following:
* for task 1:

![image](task_1_output.png)

* for task 2:

![image](task_2_output.png)

* for task 3:

![image](task_3_output.png)

### Installing
* Clone the repository in your local machine
* Open the command line interface (CLI) 
* install the required libraries using ```pip install -r requirements.txt```
* Ensure that you have a huggingface token with permission to read models
* Get a DeepL translator API key at https://www.deepl.com/en/docs-api
* Create a file called ```api_keys.py``` under the dir ```./task_3/```, the file should contain the following:
```
DEEPL_KEY = <your_deepl_api_key_as_str>
# (for gracemikaela@gmail.com a DeepL API key should be available as a canvas comment or file)
```
* run the script using  ```python run.py```

### Tests: 

To run tests, from the root dir of the repo, call: 
```
python -m pytest tests
```

## Authors

Contributors names and contact info:

Jose Angel Del Angel Dominguez  
[joseangeldelangel@gmail.com](mailto:joseangeldelangel@gmail.com)