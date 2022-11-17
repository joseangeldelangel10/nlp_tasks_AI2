'''
This code will take the first 100 lines of the UN spanish transcriptions found
on the 'europarl-v7.es-en.es' file, translate such lines using the deepl translation
API and the argostranslate API, found on:

- https://www.deepl.com/en/docs-api#:~:text=With%20the%20DeepL%20API%20Free,prioritized%20execution%20of%20translation%20requests. 
- https://github.com/argosopentech/argos-translate

Respectively.

Then the translations from both APIs will be evaluated using the BLEU score using the 
UN english translation of the spanish file 
'''

# One note on the tests you wrote: 
# the testing in general looks really good -- you have done a great job!! 

# it's better to split into more, smaller tests, so that each one tests a small piece
# of functionality, and when one fails, you know exactly where the failure is. 
# https://docs.python-guide.org/writing/tests/

# Just FYI, if tests have similar setup, you can use pytest to define a pre-test setup that
# you do for all/some tests just once, so you are not repeating work. 
# but in this case it's fine to just repeat the setup steps in each unit test, if
# that's easiest -- that's more advanced testing but wanted to send info in case you
# want to learn. 
# 

import requests
import json
from nltk.translate.bleu_score import sentence_bleu
import os

N_LINES_TO_TRANSLATE=100

def translate_to_english_using_deepl(string, api_key):
    domain_1 = 'https://api-free.deepl.com/v2/translate'
    headers_1 = {'Authorization': 'DeepL-Auth-Key {k}'.format(k=api_key)}
    data_1 = {"text":string, "target_lang":"EN"}
    response_1 = requests.post(domain_1, headers=headers_1, data=data_1)    
    translation_1 = response_1.json()["translations"][0]["text"]
    return translation_1

def translate_to_english_using_argostranslate(string):
    domain_2 = 'https://translate.argosopentech.com/translate'
    headers_2 = { "Content-Type": "application/json" }
    data_2 = {"q":string, "source":"es", "target":"en"}
    response_2 = requests.post(domain_2, headers=headers_2, data=json.dumps(data_2))    
    translation_2 = response_2.json()["translatedText"]
    return translation_2

def task_3(api_key, calling_dir = "."):
    # one note: usually the ``with open(filename, "r") as f:`` pattern is preferred for filehandling! 
    # this is becuase it auto-closes the file, here you are leaving them open since you are not calling Close
    # see https://www.programiz.com/python-programming/file-operation
    f_es = open( os.path.join( calling_dir, "es-en/europarl-v7.es-en.es") , "r")
    f_en = open( os.path.join( calling_dir, "es-en/europarl-v7.es-en.en") , "r")

    deepl_blues = []
    argostranslate_blues = []

    for _ in range(N_LINES_TO_TRANSLATE): # the _ indicates that variable is not used
        spanish_sentence = f_es.readline()
        english_reference = [f_en.readline()]
        deepl_translation = translate_to_english_using_deepl(spanish_sentence, api_key)
        argostranslate_translation = translate_to_english_using_argostranslate(spanish_sentence)
        deepl_blues.append( sentence_bleu(english_reference, deepl_translation) )
        argostranslate_blues.append( sentence_bleu(english_reference, argostranslate_translation) )    

    print("DEEPL_TRANSLATOR: {s}".format(s=sum(deepl_blues)/len(deepl_blues))) 
    # one tiny nit, if someone passes an empty file this will fail with a divide by zero error. so you could assert beforehand that these are not len 0 
    print("ARGOSTRANSLATE_TRANSLATOR: {s}".format(s=sum(argostranslate_blues)/len(argostranslate_blues)))

if __name__ == "__main__":
    from api_keys import DEEPL_KEY
    task_3(api_key=DEEPL_KEY)
