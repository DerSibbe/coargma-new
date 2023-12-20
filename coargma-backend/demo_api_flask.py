from flask import Flask, request, jsonify
import demo_api_backend_master as dabm
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def home():
    return "helloworld"

#identify_language(obj1, obj2, asp)
@app.get("/id_language")
def get_lang_id():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    return jsonify(dabm.identify_language(obj1, obj2, asp))

#translate_ru_en(obj1: str, obj2: str, asp = "")
@app.get("/translate_ru_en")
def get_translation_ru_en():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    return jsonify(dabm.translate_ru_en(obj1, obj2, asp))

#translate_en_ru(obj1: str, obj2: str, asp = "")
@app.get("/translate_en_ru")
def get_translation_en_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    return jsonify(dabm.translate_en_ru(obj1, obj2, asp))

#translate_objs_asp(obj1: str, obj2: str, asp = "")
@app.get("/translate_objs_asp")
def get_translation():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    return jsonify(dabm.translate_objs_asp(obj1, obj2, asp))

#get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp = "", altsearch = False)
@app.get("/extendedsearch_en")
def get_result_en_combinedsearch():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    altsearch = request.args.get('altsearch', '')
    if altsearch == "True":
        altsearch = True
    else:
        altsearch = False
    return jsonify(dabm.get_result_on_objs_asp_withaltsearch_en(obj1, obj2, asp, altsearch))

#get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp = "", altsearch = False)
@app.get("/extendedsearch_ru")
def get_result_ru_combinedsearch():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    altsearch = request.args.get('altsearch', '')
    if altsearch == "True":
        altsearch = True
    else:
        altsearch = False
    return jsonify(dabm.get_result_on_objs_asp_withaltsearch_ru(obj1, obj2, asp, altsearch))

#get_result_on_objs_asp_multiling(obj1, obj2, asp = "", language_id = "", altsearch = False)
@app.get("/get_result_multilingual")
def get_result_multilingual():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args.get('asp', '')
    lang_id = request.args.get('lang_id', '')
    altsearch = request.args.get('altsearch', '')
    if altsearch == "True":
        altsearch = True
    else:
        altsearch = False
    return jsonify(dabm.get_result_on_objs_asp_multiling(obj1, obj2, asp, lang_id, altsearch))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=15555, debug=True)