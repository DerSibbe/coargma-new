import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from string import punctuation
from collections import defaultdict, OrderedDict
from elasticsearch6 import Elasticsearch
import snowballstemmer
# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd
import pymorphy2
import json
from pymystem3 import Mystem
from tqdm import tqdm
from typing import Union
from collections import defaultdict
import re
from pymystem3 import Mystem
m = Mystem()
# m.analyze('хорошо')
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
pipe = pipeline("text-classification", model="lilaspourpre/rubert-tiny-stance-calssification")
stemmer = snowballstemmer.stemmer('russian')
es = Elasticsearch(["http://ltdemos.informatik.uni-hamburg.de/depcc-index"], http_auth = ("eugen", "testtest"), port=80)
assert es.ping(), "ES is not available"
good = pd.read_csv('better.txt')
bad = pd.read_csv('worse.txt')
y_param = 0.8 # @param {type:"slider", min:0, max:1, step:0.1}
sigma_score = 0.1 # @param {type:"slider", min:0, max:1, step:0.1}


big_goods = dict()
for word in tqdm(good.word.values):
    for par in morph.parse(word)[0].lexeme:
        if 'Supr' in par.tag:
            big_goods[par.word] = 'Supr'
        if 'COMP' in par.tag:
            big_goods[par.word] = 'COMP'


big_bads = dict()
for word in tqdm(bad.word.values):
    for par in morph.parse(word)[0].lexeme:
        if 'Supr' in par.tag:
            big_bads[par.word] = 'Supr'
        if 'COMP' in par.tag:
            big_bads[par.word] = 'COMP'


#for question classification
model = AutoModelForSequenceClassification.from_pretrained("lilaspourpre/rubert-tiny-comp_question_classification", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('lilaspourpre/rubert-tiny-comp_question_classification')

#for obj/asp extraction
model_checkpoint = "lilaspourpre/rubert-tiny-obj-asp"
token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

def classify_input_ru(question: str) -> bool:
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
    if predicted_class_id == 1:
        return True
    return False

# 3) create basic API for the demo, the following functions are expected:

# 3.1 given a text, extract objects and aspect:

# param: question: str; natural language comparative question
# return: tuple(str, str, str), two objects and comparative aspects of the question

def extract_objs_asp_ru(question: str):
    # 1.: Classify question as comparative or non comparative
    if classify_input_ru(question) == True:
        # 2.: If comparative, extract objects and aspects
        labeled_tokens = [[entry['entity_group'], entry['word']] for entry in token_classifier(question)]
        labeled_tokens_dict = postprocess_obj_asp(question, labeled_tokens)
        objs = [labeled_tokens_dict["Object1"], labeled_tokens_dict["Object2"]]
        if "" in objs:
            #question was falsely identified as comparative but only contains one or zero objects
            return "", "", ""
        asp = labeled_tokens_dict.get("Aspect", "")
        return objs[0], objs[1], asp
    return "", "", ""

def postprocess_obj_asp(question, model_output):
    labelled_words = {"Aspect": "", "CommonObj": "", "Object1": "", "Object2": ""}
    question_words = [i.strip(punctuation) for i in question.split()]

    for label, word_part in model_output:
        if word_part.startswith("##"):
            labelled_words[label] += word_part.strip("##")
        else:
            if labelled_words[label]=="":
                labelled_words[label] += word_part
            else:
                labelled_words[label] += " "
                labelled_words[label] += word_part
    
    final = defaultdict(OrderedDict)

    for label, words in labelled_words.items():
        for word in words.split():
            for q_word in question_words:
                if q_word.startswith(word):
                    final[label][q_word]=''
    
    final_dict = dict([(i, " ".join(j.keys())) for (i,j) in final.items()])

    return final_dict

# 3.2 given objects and aspect make a request to CAM and select 20 arguments with links (top-10 for each object) (I have code for that, here
# is the link to colab (but no links are returned there): https://colab.research.google.com/drive/1X3zflENMNFSegxM-N5IQ3z-oIeDHJ-4c?usp=sharing
# CAM is off right now, but hopefully will be up starting next week. If CAM will not be up, then just return some 20 hardcoded args with the
# random links

# param: obj1: str, obj2: str, aspect: str; two objects and one comparative aspect
# return: tuple[dict, list[tuple[str, str]]]; dict with winner data (object and percentage), list with arguments and links for both objects
def extract_data_from_CAM_ru(obj1: str, obj2: str, aspect: str, top=10) -> tuple[dict, list[tuple[str, str]]]: #to be edited
    #extract args, links, winner and percentage from CAM
    result = request_es(obj1, obj2, aspect)
    top10args_obj1 = [i['text'] for i in result['object1']['sentences'][:top]] #args
    top10args_obj2 = [i['text'] for i in result['object2']['sentences'][:top]]
    top10args_obj1_links = ["" for i in result['object1']['sentences'][:top]] #links (only first link if link is provided multiple times)
    top10args_obj2_links = ["" for i in result['object2']['sentences'][:top]]
    winner = result['winner']
    winner_points = max(result['object1']['totalPoints'], result['object2']['totalPoints'])
    total_points = result['object1']['totalPoints'] + result['object2']['totalPoints']
    if total_points == 0:
        percentage = 50.0
    else:
        percentage = round(100*winner_points/total_points, 1)
    #turn args and links into list of tuples [("arg1", "link1"), (...
    args_with_links = []
    for i in range(min(10, len(top10args_obj1))):
        args_with_links.append((top10args_obj1[i], top10args_obj1_links[i]))
    for i in range(min(10, len(top10args_obj2))):
        args_with_links.append((top10args_obj2[i], top10args_obj2_links[i]))
    return {"winner":  winner, "percentage": percentage}, args_with_links

# From CAM you can get not only objects with links, but also who is the winner and percentage, in is in the output of requests.get(address)
# in colab, just look inside :) If you do not find anything, keep it hardcoded.

# 3.3 given objects and aspect and arguments form CAM generate a text we use to put in the model (I can do it and you can return any example
# from the dataset or you can look how the input looks like and create the text yourself, it is quite easy to understand what is the template
# and where to paste input)

# param: obj1: str, obj2: str, aspect: str, arguments: list[str]; objects, comparative aspect and list of arguments for either object
# return: str; template input text as used for ChatGPT asking for a comparison of obj1 and obj2 with a list of args
def create_template_ru(obj1: str, obj2: str, aspect: str, arguments: list[str]) -> str:
    template_with_text = "Write a comparison of \""+obj1+"\" and \""+obj2+"\". Summarize only relevant arguments from the list.\n\n"\
        +"\n".join(arguments)+\
        "\n\nAfter the summary, list the arguments you used below the text. Put citations in brackets inside the text. Do not even mention "\
        "arguments that are not relevant to georgia and virginia."
    return template_with_text


# 3.4 given text generate summary (using the fine-tuned model, I have it, but let us keep it hardcoded for now, just return any summary)

# param: text: str; text asking for a comparison of two objects under a list of arguments (see create_template())
# return: str; comparison of aforementioned objects using the list of arguments
def predict_summary_ru(text: str) -> str:
    #TODO include fine-tuned model, currently only returning one hard coded summary
    summary = "Georgia and Virginia are two states in the United States, with their own unique characteristics and qualities. \n\nSome "\
        "arguments mention that Georgia is probably better than Virginia Tech [1], has a relatively freer political climate [5][9], and"\
        " has a good football team [4]. Others argue that Virginia has a nicer climate [15], lighter accents [12], and a better defense"\
        " [20]. \n\nSome comparisons are also made between different schools from the two states, such as Georgia Tech's offense being "\
        "better than Virginia Tech's defense [7], and Tech being better than FSU and Georgia Tech [11]. \n\nUltimately, it's hard to sa"\
        "y which state is \"better\" as it depends on individual preferences and experiences. Both states have their own unique history"\
        ", culture, and geography that makes them worth exploring.  \n\nArguments used: 1, 4, 5, 7, 9, 11, 12, 15, 20."
    return summary


# 3.5 correct summary and list of links: keep only those arguments, that were used in summary and reenumerate them in the summary (from 1 to n,
# where n = number_links_used)

# param: summary: str, args_with_links: list[tuple[str, str]]; Summary text comparing two objects under a list of args, and list of args
#        with corresponding links 
# return: tuple[str, list[tuple[str, str]]]; Summary with args renumbered from 1 to number of args used, and list of args with corresponding
#        links stripped of unused args
def correct_summary_and_links_ru(summary: str, args_with_links: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    improved_summary = summary
    improved_args_with_links = []
    args_used = summary.split("\n\n")[-1].replace(".", "").split(": ")[1].split(", ")
    args_used = [int(x) for x in args_used] #gives list of numbers of args used
    for i in range(len(args_used)):
        old_index = args_used[i]
        new_index = i + 1
        #replace old index with new numbering
        improved_summary = improved_summary.replace("["+str(old_index)+"]", "["+str(new_index)+"]")
        #only keep args with links of those args that were used in the summary
        if old_index-1 < len(args_with_links): #this condition may be removed once the summary generating model is implemented
            improved_args_with_links.append(args_with_links[old_index-1])
    improved_summary = improved_summary.split("\n\nArguments used: ")[0].strip() # Removes "Arguments used: x, y, z,..." from output text
    return improved_summary, improved_args_with_links


# 3.6 the whole pipeline that accepts obj1, obj2 and returns the summary, percentages and links:

def get_result_on_objs_asp_ru(obj1: str, obj2: str, aspect: str, top=10) -> tuple[str, list[tuple[str, str]], str, float]: 
    winner_data, args_with_links = extract_data_from_CAM_ru(obj1, obj2, aspect, top=10)
    arguments = [arg[0] for arg in args_with_links]
    return args_with_links, winner_data['winner'], winner_data['percentage']

 
# 3.7 the whole pipeline that accepts the question and returns the summary, percentages and links (should have two functions (3.1 and 3.6):

def get_result_on_question_ru(question: str, top: int=10) -> tuple[str, list[tuple[str, str]], str, float]:
    extracted_data = extract_objs_asp_ru(question)
    if extracted_data == ("", "", ""):
        return "Question is not comparative!"
    (obj1, obj2, aspect) = extracted_data #extract_objs_asp(question)
    return get_result_on_objs_asp_ru(obj1, obj2, aspect, top)


def index_search_elastic(hits, min_words, max_words, obj1, obj2, common_obj= "", aspect= ""):
    query_text = f"({obj1.lower()}) AND ({obj2.lower()})"
    if common_obj != "":
        query_text += f" AND ({common_obj.lower()})"
    if aspect != "":
        query_text += f" AND ({aspect.lower()})"
    es_result = es.search(
      index="ru_oscar.sentences",
      body={
          "from": 0,
          "size": hits,
          "query": {
            "bool": {
              "must": [
                {
                  "query_string": {
                    "query":query_text,
                    "fields": ["sentence"]
                  }
                }
              ],
              "filter": [
                {"range": {"n_words": {"gte": min_words, "lte": max_words}}}
              ]
            }
          }
      }
    )
    print(f"Query: {query_text}")
    print(f"Got {len(es_result['hits']['hits'])} hits")
  # print(es_result)
    es_hits = es_result["hits"]["hits"]
    clean_result = []
    for es_hit in es_hits:
        clean_result.append({
             "text": es_hit["_source"]["sentence"],
          "CommonObj" : common_obj,
          "Aspect": aspect,
          "es_max_score": es_result["hits"]["max_score"],
          "es_sent_score": es_hit["_score"]
      })
    return clean_result

def search_es(q_obj1, q_obj2, com_obj, asp):
    es_query = index_search_elastic(10000, 10, 30, q_obj1, q_obj2, com_obj, asp)
    filtered_result = list()
    test_exist_set = set()
    for query in es_query:
        if query['text'] not in test_exist_set:
            filtered_result.append((query['text'], query['es_sent_score'], query['es_max_score']))
            test_exist_set.add(query['text'])
    return filtered_result

def baseline_extract_comp(adjs: list, founded_adjectives: list[str], text: str, obj1: str, obj2: str, adj_type: str):
    for founded_adj in founded_adjectives:
        ltext = text.lower()
        splitted = ltext.split(founded_adj)
        # подходят только предложения, где прилагательное находится между двумя объектами
        if obj1 in splitted[0] and obj2 in splitted[0]:
            #print('objects before adjective', text)
            pass
        elif obj1 in splitted[1] and obj2 in splitted[1]:
            #print('objects after adjective', text)
            pass
        else:
            # теперь adj - это список найденных в тексте плохих/хороших прилагательных
            if re.search(r"(\bне).{,3}%s" % founded_adj, text): # меняем тип на противоположный если есть не перед прилагательным
                # print(founded_adj, '###', modified_text)
                if adj_type == 'pos':
                    adj_type = 'neg'
                else:
                    adj_type = 'pos'

            if adjs[founded_adj] == 'COMP':
                #print('found comparative', text, founded_adj, adj_type, 'BETTER' if adj_type == 'pos' else 'WORSE', 'obj1' if obj1 in splitted[0] else 'obj2')
                return 1, 'BETTER' if adj_type == 'pos' else 'WORSE', 'obj1' if obj1 in splitted[0] else 'obj2'
    return 0, 'NONE', 'NONE'

def baseline(texts, obj1_stem, obj2_stem):
    texts_after_baseline = []
    for text in texts:
        good_words = [word for word in big_goods.keys() if word in text[0]]
        bad_words = [word for word in big_bads.keys() if word in text[0]]
        confidence = 0
        if good_words:
            confidence, comp_type, for_object = baseline_extract_comp(big_goods, good_words, text[0], obj1_stem, obj2_stem, 'pos')
            if confidence != 0:
                texts_after_baseline.append((text[0], comp_type, for_object, text[1], text[2], confidence))
    if bad_words and confidence == 0:
        confidence, comp_type, for_object = baseline_extract_comp(big_bads, bad_words, text[0], obj1_stem, obj2_stem, 'neg')
        if confidence != 0:
            texts_after_baseline.append((text[0], comp_type, for_object, text[1], text[2], confidence))
    return texts_after_baseline

def find_for_object_for_class(sentence, obj1_stem, obj2_stem):
    if sentence.find(obj1_stem) < sentence.find(obj2_stem):
        return 'obj1'
    else:
        return 'obj2'

def bert(sentences_with_scores, obj1_stem, obj2_stem):
    texts_to_class = [x[0] for x in sentences_with_scores]
    class_result = pipe(texts_to_class)
    class_list_of_sentences_with_scores = []
    for text_idx in range(len(texts_to_class)):
        if class_result[text_idx]['label'] != 'NONE':
            class_list_of_sentences_with_scores.append((texts_to_class[text_idx], class_result[text_idx]['label'], find_for_object_for_class(texts_to_class[text_idx]), sentences_with_scores[text_idx][1], sentences_with_scores[text_idx][2], class_result[text_idx]['score']))
    return class_list_of_sentences_with_scores

def classify_sentences_baseline(list_of_sentences_with_scores, obj1_stem, obj2_stem):
    list_of_sentences_with_all_scores_better_worse_only = baseline(list_of_sentences_with_scores, obj1_stem, obj2_stem) # предложение, тип сравнения, какой объект, скор, макс_скор, уверенность
    return list_of_sentences_with_all_scores_better_worse_only

def classify_sentences_bert(list_of_sentences, obj1_stem, obj2_stem):
    list_of_sentences_with_all_scores_better_worse_only = bert(list_of_sentences_with_scores, obj1_stem, obj2_stem) # предложение, тип сравнения, какой объект, скор, макс_скор, уверенность
    return list_of_sentences_with_all_scores_better_worse_only

def calculate_bert_score(es_score, max_es_score, classifier_score):
    if classifier_score >= y_param:
        return es_score + max_es_score
    else:
        return es_score * sigma_score


def calculate_baseline_score(es_score, max_es_score):
    return es_score + max_es_score


def calculate_final_scores_bert(list_of_sentences_with_scores):
    sentences_with_final_scores_obj1 = []
    sentences_with_final_scores_obj2 = []
    for sent, comp_type, comp_obj, es_score, es_max_score, conf in list_of_sentences_with_scores:
        if (comp_obj == 'obj1' and comp_type == 'BETTER') or (comp_obj == 'obj2' and comp_type == 'WORSE'):
            sentences_with_final_scores_obj1.append((sent, calculate_bert_score(es_score, es_max_score, conf)))
        else:
            sentences_with_final_scores_obj2.append((sent, calculate_bert_score(es_score, es_max_score, conf)))
    sorted_sentences_with_final_scores_obj1 = sorted(sentences_with_final_scores_obj1, key=lambda x: x[1], reverse=True)
    sorted_sentences_with_final_scores_obj2 = sorted(sentences_with_final_scores_obj2, key=lambda x: x[1], reverse=True)
    return sorted_sentences_with_final_scores_obj1, sorted_sentences_with_final_scores_obj2


def calculate_final_scores_baseline(list_of_sentences_with_scores):
    sentences_with_final_scores_obj1 = []
    sentences_with_final_scores_obj2 = []
    for sent, comp_type, comp_obj, es_score, es_max_score, conf in list_of_sentences_with_scores:
        if (comp_obj == 'obj1' and comp_type == 'BETTER') or (comp_obj == 'obj2' and comp_type == 'WORSE'):
          sentences_with_final_scores_obj1.append((sent, calculate_baseline_score(es_score, es_max_score)))
        else:
          sentences_with_final_scores_obj2.append((sent, calculate_baseline_score(es_score, es_max_score)))
    return sentences_with_final_scores_obj1, sentences_with_final_scores_obj2

def ru_cam(baseline, list_of_sentences_with_scores, obj1_stem, obj2_stem):
    if baseline:
        print("Classify using baseline")
        res_base = classify_sentences_baseline(list_of_sentences_with_scores, obj1_stem, obj2_stem)
        print(f"baseline len: {len(res_base)}")
        result_obj1, result_obj2 = calculate_final_scores_baseline(res_base)
        print(f"for obj1: {len(result_obj1)} | for obj2: {len(result_obj2)}")
        return result_obj1, result_obj2, res_base
    else:
        print("Classify using bert")
        res_bert = classify_sentences_bert(list_of_sentences_with_scores, obj1_stem, obj2_stem)
        print(f"bert len: {len(res_bert)}")
        result_obj1, result_obj2 = calculate_final_scores_bert(res_bert)
        print(f"for obj1: {len(result_obj1)} | for obj2: {len(result_obj2)}")
        return result_obj1, result_obj2, res_bert


def jsonify_sentences(obj, result_obj):
    sentences = []
    for text, cam_score in result_obj:
        sentences.append({"text": text, "CAM_score": cam_score})
    return sentences

def select_winner(obj1, obj2, result_obj1, result_obj2):
    total_points_a = sum([j for i,j in result_obj1])
    total_points_b = sum([j for i,j in result_obj2])
    winner = obj1 if total_points_a >= total_points_b else obj2
    return total_points_a, total_points_b, winner


def request_es(obj1: str, obj2: str, aspect: str, common:str="") -> tuple[dict, list[tuple[str, str]]]:
    obj1_stem = stemmer.stemWords([obj1])[0]
    obj2_stem = stemmer.stemWords([obj2])[0]
    use_baseline = True # @param {type:"boolean"}
    list_of_sentences_with_scores = search_es(obj1, obj2, common, aspect) # это отсортированный список пар: (предложение, скор, макс_скор)
    print(f"filtered len: {len(list_of_sentences_with_scores)}")
    result_obj1, result_obj2, class_report = ru_cam(use_baseline, list_of_sentences_with_scores, obj1_stem, obj2_stem)
    total_points_a, total_points_b, winner = select_winner(obj1, obj2, result_obj1, result_obj2)
    return {
         "object1": {"name": obj1, "sentences": jsonify_sentences(obj1, result_obj1[0:10]), "totalPoints":total_points_a}, 
         "object2": {"name": obj2, "sentences": jsonify_sentences(obj2, result_obj2[0:10]), "totalPoints":total_points_b}, 
         "winner": winner
    }
