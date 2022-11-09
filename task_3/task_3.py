import requests
import json
from nltk.translate.bleu_score import sentence_bleu
from api_keys import DEEPL_KEY

def translate_to_english_using_deepl(string):
    domain_1 = 'https://api-free.deepl.com/v2/translate'
    headers_1 = {'Authorization': 'DeepL-Auth-Key {k}'.format(k=DEEPL_KEY)}
    data_1 = {"text":string, "target_lang":"EN"}
    response_1 = requests.post(domain_1, headers=headers_1, data=data_1)
    #print(response_1.json())
    translation_1 = response_1.json()["translations"][0]["text"]
    return translation_1

def translate_to_english_using_argostranslate(string):
    domain_2 = 'https://translate.argosopentech.com/translate'
    headers_2 = { "Content-Type": "application/json" }
    data_2 = {"q":string, "source":"es", "target":"en"}
    response_2 = requests.post(domain_2, headers=headers_2, data=json.dumps(data_2))
    #print(response_2.json())
    translation_2 = response_2.json()["translatedText"]
    return translation_2

f_es = open("es-en/europarl-v7.es-en.es", "r")
f_en = open("es-en/europarl-v7.es-en.en", "r")

maximum_iter = 100
i = 1
deepl_blues = []
argostranslate_blues = []

while i <= maximum_iter:
    '''
    spanish_sentence = f_es.readline()
    english_reference = [f_en.readline()]
    deepl_translation = translate_to_english_using_deepl(spanish_sentence)
    argostranslate_translation = translate_to_english_using_argostranslate(spanish_sentence)
    deepl_blues.append( sentence_bleu(english_reference, deepl_translation) )
    argostranslate_blues.append( sentence_bleu(english_reference, argostranslate_translation) )
    '''
    i += 1

#print("DEEPL_TRANSLATOR: {s}".format(s=sum(deepl_blues)/len(deepl_blues)))
#print("ARGOSTRANSLATE_TRANSLATOR: {s}".format(s=sum(argostranslate_blues)/len(argostranslate_blues)))