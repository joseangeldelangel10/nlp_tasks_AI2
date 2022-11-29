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
import requests
import json
from nltk.translate.bleu_score import sentence_bleu
import os

N_LINES_TO_TRANSLATE = 10

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
    global N_LINES_TO_TRANSLATE
    f_es = open( os.path.join( calling_dir, "es-en/europarl-v7.es-en.es") , "r")
    f_en = open( os.path.join( calling_dir, "es-en/europarl-v7.es-en.en") , "r")
    
    deepl_blues = []
    argostranslate_blues = []

    for _ in range(N_LINES_TO_TRANSLATE):    
        spanish_sentence = f_es.readline()
        english_reference = [f_en.readline()]
        deepl_translation = translate_to_english_using_deepl(spanish_sentence, api_key)
        argostranslate_translation = translate_to_english_using_argostranslate(spanish_sentence)
        deepl_blues.append( sentence_bleu(english_reference, deepl_translation) )
        argostranslate_blues.append( sentence_bleu(english_reference, argostranslate_translation) )            

    f_es.close()
    f_en.close()

    if len(deepl_blues) > 0 and len(argostranslate_blues) > 0:
        print("DEEPL_TRANSLATOR: {s}".format(s=sum(deepl_blues)/len(deepl_blues)))
        print("ARGOSTRANSLATE_TRANSLATOR: {s}".format(s=sum(argostranslate_blues)/len(argostranslate_blues)))
    else:
        print("No lines given for translation, please check if the value of N_LINES_TO_TRANSLATE is correct")

if __name__ == "__main__":
    from api_keys import DEEPL_KEY
    task_3(api_key=DEEPL_KEY)
