from flask import Flask, request, jsonify
import demo_api_backend as dab

app = Flask(__name__)

@app.route("/")
def home():
    return "helloworld"

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=15555, debug=True)