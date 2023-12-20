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
def translate_ru_en(obj1: str, obj2: str, asp = ""):
    input = "; ".join([element for element in [obj1, obj2, asp] if element != ""])
    translation = pipe_ru_en(input)[0]['translation_text']
    translated_objs = translation.split("; ")
    return translated_objs

def translate_en_ru(obj1: str, obj2: str, asp = ""):
    input = "; ".join([element for element in [obj1, obj2, asp] if element != ""])
    translation = pipe_en_ru(input)[0]['translation_text']
    translated_objs = translation.split("; ")
    return translated_objs

# translate objects between english and russian
def translate_objs_asp(obj1: str, obj2: str, asp = ""):
    src_language_id = identify_language(obj1, obj2, asp)
    if src_language_id == "ru":
        return translate_ru_en(obj1, obj2, asp)
    return translate_en_ru(obj1, obj2, asp)


# search only in given language if altsearch == False; translate and search in both english and russian if altsearch == True
def get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp = "", altsearch = False):
    if altsearch == True:
        translation_list = translate_en_ru(obj1, obj2, asp)
        return dab_ru.get_result_on_objs_asp_ru(*translation_list), dab_en.get_result_on_objs_asp(obj1, obj2, asp) 
    return dab_en.get_result_on_objs_asp(obj1, obj2, asp)

def get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp = "", altsearch = False):
    if altsearch == True:
        translation_list = translate_ru_en(obj1, obj2, asp)
        return dab_en.get_result_on_objs_asp(*translation_list), dab_ru.get_result_on_objs_asp_ru(obj1, obj2, asp)
    return dab_ru.get_result_on_objs_asp_ru(obj1, obj2, asp)

# language_id: "en" or "ru" if user provides language for input question, otherwise ""
# altsearch: if True, do search through both dab_en and dab_ru, translate input as necessary
# TODO translate output into input language?
def get_result_on_objs_asp_multiling(obj1, obj2, asp = "", language_id = "", altsearch = False):
    lang_id = language_id
    if language_id not in ["en", "ru"]:
        lang_id = identify_language(obj1, obj2, asp)
    if lang_id == "ru":
        return get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp, altsearch)
    return get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp, altsearch)