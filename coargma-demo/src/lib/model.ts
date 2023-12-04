import { BehaviorSubject, combineLatest, map } from "rxjs";

const distribution = new BehaviorSubject<{object1: number, object2: number} | null>(null);

const summary = new BehaviorSubject<string[]>([]);

const sources_obj1 = new BehaviorSubject<{url: string, caption: string}[]>([]);

const sources_obj2 = new BehaviorSubject<{url: string, caption: string}[]>([]);


const question = new BehaviorSubject("");


function setQuestion(value: string) {
    question.next(value);
}

const object1 = new BehaviorSubject("");

function setObject1(value: string) {
    object1.next(value);
}

const object2 = new BehaviorSubject("");

function setObject2(value: string) {
    object2.next(value);
}

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

const use_baseline = new BehaviorSubject<boolean>(true);

function setBaselineUse(value: boolean) {
    use_baseline.next(value);
    console.log(use_baseline.value)
}

const couldCompare =
    combineLatest([question, object1, object2, aspect])
        .pipe(map(([question, object1, object2, aspect]) =>
            question || object1 && object2 && aspect));


async function RequestCompare() {
    const response = await fetch("http://localhost:15555/get_result_on_question_ru?question=" + encodeURIComponent(question.value) + "&top=" + encodeURIComponent(numberOfArguments.value) + "&use_baseline=" + encodeURIComponent(use_baseline.value));
    console.log(response)
    const result = await response.json();
    return result
}

function compare() {
    console.log(question.value);
    RequestCompare().then(result => {
        console.log(result);
        distribution.next({
            object1: result.percentage_winner,
            object2: 100-result.percentage_winner
        });
        summary.next([""]);
        sources_obj1.next(Object.values(result.args.object1.sentences).map((value) => ({
            url: String(value.link),
            caption: String(value.text)
          })));

        sources_obj2.next(Object.values(result.args.object2.sentences).map((value) => ({
            url: String(value.link),
            caption: String(value.text)
          })));


    })
}

export default {
    distribution,
    summary,
    sources_obj1,
    sources_obj2,
    question,
    setQuestion,
    object1,
    setObject1,
    object2,
    setObject2,
    aspect,
    setAspect,
    numberOfArguments,
    use_baseline,
    setNumberOfArguments,
    setBaselineUse,
    couldCompare,
    compare
};
