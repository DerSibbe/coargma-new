import { BehaviorSubject, combineLatest, map } from "rxjs";

const distribution = new BehaviorSubject<{object1: number, object2: number} | null>(null);

const summary = new BehaviorSubject<string[]>([]);

const sources_obj1 = new BehaviorSubject<{url: string, caption: string}[]>([]);

const sources_obj2 = new BehaviorSubject<{url: string, caption: string}[]>([]);


const question = new BehaviorSubject("Кто лучше: кошки или собаки?");


function setQuestion(value: string) {
    question.next(value);
}

function isQuestionEmpty() {
    return question.value === ""; 
}

const object1 = new BehaviorSubject("");

function setObject1(value: string) {
    object1.next(value);
}

const object2 = new BehaviorSubject("");

function setObject2(value: string) {
    object2.next(value);
}

const compare_object1 = new BehaviorSubject("");

const compare_object2 = new BehaviorSubject("");

const aspect = new BehaviorSubject("");

function setAspect(value: string) {
    aspect.next(value);
}

const numberOfArguments = new BehaviorSubject<number>(10);

function setNumberOfArguments(value: number) {
    if (value < 1) {
        value = 1;
    }
    numberOfArguments.next(value);
    console.log(numberOfArguments.value)
}

const use_baseline = new BehaviorSubject<boolean>(false);

function setBaselineUse(value: boolean) {
    use_baseline.next(value);
    console.log(use_baseline.value)
}

const couldCompare =
    combineLatest([question, object1, object2, aspect])
        .pipe(map(([question, object1, object2, aspect]) =>
            question || object1 && object2));

const isLoading = new BehaviorSubject<boolean>(false);

const backendHostURL = 'http://localhost:15555'

async function RequestCompareQuestion() {
    try {
        const response = await fetch(backendHostURL + "/get_result_on_question_ru?question=" + encodeURIComponent(question.value) + "&top=" + encodeURIComponent(numberOfArguments.value) + "&use_baseline=" + encodeURIComponent(use_baseline.value));
        console.log(response)
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('An error occurred:', error);
        isLoading.next(false);
    }
}

async function RequestCompareObjAsp() {
    try {
        const response = await fetch(backendHostURL + "/get_result_on_objs_asp_ru?obj1=" + encodeURIComponent(object1.value) + "&obj2=" + encodeURIComponent(object2.value) + "&asp=" + encodeURIComponent(aspect.value) + "&top=" + encodeURIComponent(numberOfArguments.value) + "&use_baseline=" + encodeURIComponent(use_baseline.value));
        console.log(response)
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('An error occurred:', error);
        isLoading.next(false);
    }
}


function compare_question() {
    isLoading.next(true);
    console.log(question.value);
    RequestCompareQuestion().then(result => {
        console.log(result);
        distribution.next({
            object1: result.percentage_winner,
            object2: 100-result.percentage_winner
        });
        summary.next([""]);
        sources_obj1.next(result.args.object1.sentences.map((value: {link: string, text: string}) => ({
            url: String(value.link),
            caption: String(value.text)
        })));
        
        sources_obj2.next(result.args.object2.sentences.map((value: {link: string, text: string}) => ({
            url: String(value.link),
            caption: String(value.text)
        })));
        object1.next(result.args.object1.name);
        object2.next(result.args.object2.name);
        compare_object1.next(result.args.object1.name);
        compare_object2.next(result.args.object2.name);
        isLoading.next(false);
    })
}

function compare_obj_asp() {
    isLoading.next(true);
    console.log(`object1: ${object1.value}, object2: ${object2.value}, aspect: ${aspect.value}`);
    RequestCompareObjAsp().then(result => {
        console.log(result);
        distribution.next({
            object1: result.percentage_winner,
            object2: 100-result.percentage_winner
        });
        summary.next([""]);
        sources_obj1.next(result.args.object1.sentences.map((value: {link: string, text: string}) => ({
            url: String(value.link),
            caption: String(value.text)
        })));
        
        sources_obj2.next(result.args.object2.sentences.map((value: {link: string, text: string}) => ({
            url: String(value.link),
            caption: String(value.text)
        })));
        object1.next(result.args.object1.name);
        object2.next(result.args.object2.name);
        compare_object1.next(result.args.object1.name);
        compare_object2.next(result.args.object2.name);
        isLoading.next(false);
    })
}

export default {
    distribution,
    summary,
    sources_obj1,
    sources_obj2,
    question,
    setQuestion,
    isQuestionEmpty,
    object1,
    setObject1,
    object2,
    setObject2,
    compare_object1,
    compare_object2,
    aspect,
    setAspect,
    numberOfArguments,
    use_baseline,
    setNumberOfArguments,
    setBaselineUse,
    couldCompare,
    isLoading,
    compare_question,
    compare_obj_asp
};
