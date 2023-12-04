from flask import Flask, request, jsonify
import demo_api_backend as dab
import demo_api_backend_ru as dab_ru
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def home():
    return "helloworld"

#english methods (right now russian models are used for both as default though)

@app.get("/check_comparative")
def check_comparative():
    question = request.args['question']
    return jsonify(dab.classify_input(question))

@app.get("/extract_objs_asp")
def get_objs_asp():
    question = request.args['question']
    return jsonify(dab.extract_objs_asp(question))

@app.get("/extract_data_from_CAM")
def get_CAM_data():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    return jsonify(dab.extract_data_from_CAM(obj1, obj2, asp))

@app.get("/create_template")
def get_template():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab.extract_data_from_CAM(obj1, obj2, asp)[1]
    return jsonify(dab.create_template(obj1, obj2, asp, args_with_links))

@app.get("/predict_summary")
def get_summary():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab.extract_data_from_CAM(obj1, obj2, asp)[1]
    template = dab.create_template(obj1, obj2, asp, args_with_links)
    return jsonify(dab.predict_summary(template))

@app.get("/correct_summary_and_links")
def get_correction():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab.extract_data_from_CAM(obj1, obj2, asp)[1]
    template = dab.create_template(obj1, obj2, asp, args_with_links)
    summary = dab.predict_summary(template)
    return jsonify(dab.correct_summary_and_links(summary, args_with_links))

@app.get("/get_result_on_objs_asp")
def get_result_obj_asp():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    return jsonify(dab.get_result_on_objs_asp(obj1, obj2, asp))

@app.get("/get_result_on_question")
def get_result():
    question = request.args['question']
    return jsonify(dab.get_result_on_question(question))

#russian methods

@app.get("/check_comparative_ru")
def check_comparative_ru():
    question = request.args['question']
    return jsonify(dab_ru.classify_input_ru(question))

@app.get("/extract_objs_asp_ru")
def get_objs_asp_ru():
    question = request.args['question']
    return jsonify(dab_ru.extract_objs_asp_ru(question))

@app.get("/extract_data_from_CAM_ru")
def get_CAM_data_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    return jsonify(dab_ru.extract_data_from_CAM_ru(obj1, obj2, asp))

@app.get("/create_template_ru")
def get_template_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab_ru.extract_data_from_CAM_ru(obj1, obj2, asp)[1]
    return jsonify(dab_ru.create_template_ru(obj1, obj2, asp, args_with_links))

@app.get("/predict_summary_ru")
def get_summary_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab_ru.extract_data_from_CAM_ru(obj1, obj2, asp)[1]
    template = dab_ru.create_template_ru(obj1, obj2, asp, args_with_links)
    return jsonify(dab_ru.predict_summary_ru(template))

@app.get("/correct_summary_and_links_ru")
def get_correction_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    args_with_links = dab_ru.extract_data_from_CAM_ru(obj1, obj2, asp)[1]
    template = dab_ru.create_template_ru(obj1, obj2, asp, args_with_links)
    summary = dab_ru.predict_summary_ru(template)
    return jsonify(dab_ru.correct_summary_and_links_ru(summary, args_with_links))

@app.get("/get_result_on_objs_asp_ru")
def get_result_obj_asp_ru():
    obj1 = request.args['obj1']
    obj2 = request.args['obj2']
    asp = request.args['asp']
    top = request.args['top']
    use_baseline = request.args['use_baseline']
    return jsonify(dab_ru.get_result_on_objs_asp_ru(obj1, obj2, asp, top, use_baseline))

@app.get("/get_result_on_question_ru")
def get_result_ru():
    question = request.args.get('question')
    top = int(request.args.get('top', default=10))
    use_baseline = request.args.get('use_baseline', default=True)
    return jsonify(dab_ru.get_result_on_question_ru(question, top, use_baseline))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=15555, debug=True)