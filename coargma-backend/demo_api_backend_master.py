import demo_api_backend_en as dab_en
import demo_api_backend_ru as dab_ru
from transformers import pipeline

#for language classification (might need different model because this supports way more langs than just russian or english)
langid_pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

#for translation english to russian
pipe_en_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
#for translation russian to english
pipe_ru_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

# identify whether input question is russian or english
def identify_language(obj1: str, obj2: str, asp = ""):
    listed_args = ", ".join([element for element in [obj1, obj2, asp] if element != ""])
    translation = langid_pipe(listed_args)
    langid = translation[0]['label']
    return langid

# put objs+asp in string of format "obj1, obj2, asp" and translate from russian to english or vice versa.
# return:
#   translated_objs: list of two strings, being translations of obj1 and obj2 respectively
def translateinp_ru_en(obj1: str, obj2: str, asp = ""):
    input = "; ".join([element for element in [obj1, obj2, asp] if element != ""])
    translation = pipe_ru_en(input)[0]['translation_text']
    translated_objs = translation.split("; ")
    return translated_objs

def translateinp_en_ru(obj1: str, obj2: str, asp = ""):
    input = "; ".join([element for element in [obj1, obj2, asp] if element != ""])
    translation = pipe_en_ru(input)[0]['translation_text']
    translated_objs = translation.split("; ")
    return translated_objs

# translate objects between english and russian
def translate_objs_asp(obj1: str, obj2: str, asp = ""):
    src_language_id = identify_language(obj1, obj2, asp)
    if src_language_id == "ru":
        return translateinp_ru_en(obj1, obj2, asp)
    return translateinp_en_ru(obj1, obj2, asp)

# output format:
#  {'args': {
#       'object1': {
#           'name': str,
#           'sentences': [
#               {
#                   CAM_score': float,
#                   'link': str,
#                   'text': str
#               } *10
#           ],
#           'totalPoints': float
#       },
#       'object2': {
#           'name': str,
#           'sentences': [
#               {
#                   CAM_score': float,
#                   'link': str,
#                   'text': str
#               } *10
#           ],
#           'totalPoints': float
#        },
#   'looser': str,
#   'percentage_winner': int,
#   'winner': str
#   }
def translateout(output, srclangid: str, obj1: str, obj2: str, asp: str):
    out_obj1 = output['args']['object1']['name']
    out_obj1_tp = output['args']['object1']['totalPoints']
    out_obj2_tp = output['args']['object2']['totalPoints']
    out_winner = output['winner']
    out_perc = output['percentage_winner']
    trans_winner = obj1 if out_obj1 == out_winner else obj2
    trans_looser = obj1 if trans_winner == obj2 else obj2
    trans_pipe = pipe_en_ru if srclangid == "ru" else pipe_ru_en

    trans_obj1_sentences = []
    for entry in output['args']['object1']['sentences']:
        entry_text, entry_CAM_score, entry_link = list(entry.values())
        trans_text = trans_pipe(entry_text)[0]['translation_text']
        trans_obj1_sentences.append({'text': trans_text, 'CAM_score': entry_CAM_score, 'link': entry_link})

    trans_obj2_sentences = []
    for entry in output['args']['object2']['sentences']:
        entry_text, entry_CAM_score, entry_link = list(entry.values())
        trans_text = trans_pipe(entry_text)[0]['translation_text']
        trans_obj2_sentences.append({'text': trans_text, 'CAM_score': entry_CAM_score, 'link': entry_link})

    translated_output = {'args': {
                            'object1': {'name': obj1, 'sentences': trans_obj1_sentences, 'totalPoints': out_obj1_tp},
                            'object2': {'name': obj2, 'sentences': trans_obj2_sentences, 'totalPoints': out_obj2_tp},
                            'looser': trans_looser,
                            'percentage_winner': out_perc,
                            'winner': trans_winner}}
    return translated_output
    
# import demo_api_backend_master as dab
# print(dab.get_result_on_objs_asp_multiling("cat", "dog", altsearch=True))
# print(dab.get_result_on_objs_asp_multiling("кошки", "собаки", altsearch=True))

# search only in given language if altsearch == False; translate and search in both english and russian if altsearch == True
def get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp = "", altsearch = False):
    if altsearch == True:
        translation_list = translate_objs_asp(obj1, obj2, asp)
        return translateout(dab_ru.get_result_on_objs_asp_ru(*translation_list), "en", obj1, obj2, asp), dab_en.get_result_on_objs_asp(obj1, obj2, asp) 
    return dab_en.get_result_on_objs_asp(obj1, obj2, asp)

def get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp = "", altsearch = False):
    if altsearch == True:
        translation_list = translate_objs_asp(obj1, obj2, asp)
        return translateout(dab_en.get_result_on_objs_asp(*translation_list), "ru", obj1, obj2, asp), dab_ru.get_result_on_objs_asp_ru(obj1, obj2, asp)
    return dab_ru.get_result_on_objs_asp_ru(obj1, obj2, asp)

# language_id: "en" or "ru" if user provides language for input question, otherwise ""
# altsearch: if True, do search through both dab_en and dab_ru, translate input as necessary
def get_result_on_objs_asp_multiling(obj1, obj2, asp = "", language_id = "", altsearch = False):
    lang_id = language_id
    if language_id not in ["en", "ru"]: # ==> language_id = probably ""
        lang_id = identify_language(obj1, obj2, asp)
    if lang_id == "ru":
        return get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp, altsearch)
    return get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp, altsearch)