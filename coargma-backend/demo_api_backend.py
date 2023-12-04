import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

#for question classification
model = AutoModelForSequenceClassification.from_pretrained("lilaspourpre/rubert-tiny-comp_question_classification", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('lilaspourpre/rubert-tiny-comp_question_classification')

#for obj/asp extraction
model_checkpoint = "lilaspourpre/rubert-tiny-obj-asp"
token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

#english, keeping russian as default for now
#for question classification
#model = AutoModelForSequenceClassification.from_pretrained("uhhlt/roberta-binary-classifier", num_labels=2)
#tokenizer = AutoTokenizer.from_pretrained('uhhlt/roberta-binary-classifier')
#for obj/asp extraction
#model_checkpoint = "uhhlt/model-obj-asp-en"
#token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

def classify_input(question: str) -> bool:
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
def extract_objs_asp(question: str):# -> tuple(str, str, str):
    # 1.: Classify question as comparative or non comparative
    if classify_input(question) == True:
        # 2.: If comparative, extract objects and aspects
        labeled_tokens = [[entry['entity_group'], entry['word']] for entry in token_classifier(question)]
        objs = [entry[1].strip() for entry in labeled_tokens if entry[0] == "OBJ"]
        if len(objs) < 2:
            #question was falsely identified as comparative but only contains one or zero objects
            return "", "", ""
        if len(objs) > 2:
            #too many objects, for the sake of this task only take the first two
            objs = objs[0:2]
        asp = [entry[1] for entry in labeled_tokens if entry[0] in ["PRED", "ASP"]][0].strip()
        #problem with asp: things like "more stable" are split into "PRED more" and "ASP stable" --> asp = "more"
        #maybe solution:
        if asp == 'more':
            asp = 'more' + [entry[1] for entry in labeled_tokens if entry[0] in ["PRED", "ASP"]][1]    
        return objs[0], objs[1], asp
    return "", "", ""

# 3.2 given objects and aspect make a request to CAM and select 20 arguments with links (top-10 for each object) (I have code for that, here
# is the link to colab (but no links are returned there): https://colab.research.google.com/drive/1X3zflENMNFSegxM-N5IQ3z-oIeDHJ-4c?usp=sharing
# CAM is off right now, but hopefully will be up starting next week. If CAM will not be up, then just return some 20 hardcoded args with the
# random links

# param: obj1: str, obj2: str, aspect: str; two objects and one comparative aspect
# return: tuple[dict, list[tuple[str, str]]]; dict with winner data (object and percentage), list with arguments and links for both objects
def extract_data_from_CAM(obj1: str, obj2: str, aspect: str) -> tuple[dict, list[tuple[str, str]]]:
    lt = "ltdemos.informatik.uni-hamburg.de"
    if aspect == "":
        address = f"http://{lt}/cam-api/cam?model=default&fs=false&objectA={obj1}&objectB={obj2}"
    else:    
        address = f"http://{lt}/cam-api/cam?model=default&fs=false&objectA={obj1}&objectB={obj2}&aspect1={aspect}&weight1=5"
    x = requests.get(address)
    result = x.json()
    #extract args, links, winner and percentage from CAM
    top10args_obj1 = [i['text'] for i in result['object1']['sentences'][:10]] #args
    top10args_obj2 = [i['text'] for i in result['object2']['sentences'][:10]]
    top10args_obj1_links = [list(i['id_pair'].keys())[0] for i in result['object1']['sentences'][:10]] #links (only first link if link is provided multiple times)
    top10args_obj2_links = [list(i['id_pair'].keys())[0] for i in result['object2']['sentences'][:10]]
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
def create_template(obj1: str, obj2: str, aspect: str, arguments: list[str]) -> str:
    template_with_text = "Write a comparison of \""+obj1+"\" and \""+obj2+"\". Summarize only relevant arguments from the list.\n\n"\
        +"\n".join(arguments)+\
        "\n\nAfter the summary, list the arguments you used below the text. Put citations in brackets inside the text. Do not even mention "\
        "arguments that are not relevant to georgia and virginia."
    return template_with_text


# 3.4 given text generate summary (using the fine-tuned model, I have it, but let us keep it hardcoded for now, just return any summary)

# param: text: str; text asking for a comparison of two objects under a list of arguments (see create_template())
# return: str; comparison of aforementioned objects using the list of arguments
def predict_summary(text: str) -> str:
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
def correct_summary_and_links(summary: str, args_with_links: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
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

def get_result_on_objs_asp(obj1: str, obj2: str, aspect: str) -> tuple[str, list[tuple[str, str]], str, float]: 
    winner_data, args_with_links = extract_data_from_CAM(obj1, obj2, aspect)
    arguments = [arg[0] for arg in args_with_links]
    template = create_template(obj1, obj2, aspect, arguments)
    summary = predict_summary(template)
    summary, args_with_links = correct_summary_and_links(summary, args_with_links)
    return summary, args_with_links, winner_data['winner'], winner_data['percentage']

 
# 3.7 the whole pipeline that accepts the question and returns the summary, percentages and links (should have two functions (3.1 and 3.6):

def get_result_on_question(question: str) -> tuple[str, list[tuple[str, str]], str, float]:
    extracted_data = extract_objs_asp(question)
    if extracted_data == ("", "", ""):
        return "Question is not comparative!"
    (obj1, obj2, aspect) = extracted_data #extract_objs_asp(question)
    return get_result_on_objs_asp(obj1, obj2, aspect)