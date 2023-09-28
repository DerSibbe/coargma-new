import { BehaviorSubject, combineLatest, map } from "rxjs";

const distribution = new BehaviorSubject<{object1: number, object2: number} | null>(null);

const summary = new BehaviorSubject<string[]>([]);

const sources = new BehaviorSubject<{url: string, caption: string}[]>([]);

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

const numberOfArguments = new BehaviorSubject(10);

function setNumberOfArguments(value: number) {
    if (value < 1) {
        value = 1;
    }
    numberOfArguments.next(value);
}

const couldCompare =
    combineLatest([question, object1, object2, aspect])
        .pipe(map(([question, object1, object2, aspect]) =>
            question || object1 && object2 && aspect));

function compare() {
    distribution.next({
        object1: 60,
        object2: 40
    });
    summary.next([
        "Summary is very good[1]",
        "We should continue to do a good job[2]"
    ]);
    sources.next([
        {
          url: "https://google.com",
          caption: "Google"
        },
        {
          url: "https://github.com",
          caption: "Github"
        }
    ]);
}

export default {
    distribution,
    summary,
    sources,
    question,
    setQuestion,
    object1,
    setObject1,
    object2,
    setObject2,
    aspect,
    setAspect,
    numberOfArguments,
    setNumberOfArguments,
    couldCompare,
    compare
};
