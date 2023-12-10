import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import FloatingLabel from 'react-bootstrap/FloatingLabel';
import Model from '../../lib/model';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const question = useObservable(() => Model.question) || "";
    const object1 = useObservable(() => Model.object1) || "";
    const object2 = useObservable(() => Model.object2) || "";
    const aspect = useObservable(() => Model.aspect) || "";
    const numberOfArguments = useObservable(() => Model.numberOfArguments) || "";
    const couldCompare = useObservable(() => Model.couldCompare) || false;
    const isLoading = useObservable(() => Model.isLoading) || false;
    const useBaselineCheck = useObservable(() => Model.use_baseline) || false;


    return (<Form>
        <Form.Group className="mb-2 ">
            <FloatingLabel label="Enter your question here..." className='form-floating-attr'>
            <Form.Control type="text" className='question-attr-text'
                placeholder="Enter your question here..." value={question} onChange={e => {Model.object1.next('');Model.object2.next('');Model.aspect.next('');Model.setQuestion(e.target.value)}} />
            </FloatingLabel>
        </Form.Group>
        
        <Form.Group className="mb-2">
            <Form.Text className='text-light fw-bold'>or enter two objects for comparison</Form.Text>
        </Form.Group>

        <Form.Group as={Row} className="mb-2">
            <Col>
                <FloatingLabel label="Object 1" className='form-floating-attr'>
                    <Form.Control type="text" className='question-attr-text'
                        placeholder="Object 1" value={object1} onChange={e => {Model.setObject1(e.target.value); Model.question.next('');}}/>
                </FloatingLabel>
            </Col>
            <Col>
                <FloatingLabel label="Object 2" className='form-floating-attr'>
                    <Form.Control type="text" className='question-attr-text'
                        placeholder="Object 2" value={object2} onChange={e => {Model.setObject2(e.target.value); Model.question.next('');}}/>
                </FloatingLabel>
            </Col>
        </Form.Group>


        <Form.Group as={Row} className="mb-3">
            <Col>
                <FloatingLabel label="Aspect" className='form-floating-attr'>
                    <Form.Control type="text" className='question-attr-text'
                        placeholder="Aspect" value={aspect} onChange={e => {Model.setAspect(e.target.value); Model.question.next('');}}/>
                </FloatingLabel>
            </Col>
            <Col></Col>
        </Form.Group>

        <Form.Group as={Row} className="mb-2 d-flex justify-content-center">
            <Col style={{ display: 'flex', alignItems: 'center'}}>
                <Form.Label className='text-light fw-bold fs-5' style={{marginTop: '5px'}}>Max number of arguments</Form.Label>
            </Col>
            <Col style={{ display: 'flex', alignItems: 'center' }}>
            <Form.Control type="number" className='question-attr-text'
                        value={numberOfArguments} 
                        onChange={e => Model.setNumberOfArguments(+e.target.value)} />
            </Col>
            <Col style={{ display: 'flex', alignItems: 'center' }}>
                <Form.Check type="checkbox" defaultChecked={useBaselineCheck}
                onChange={e => Model.setBaselineUse(e.target.checked)}
                label="Use baseline"
                className='text-light fw-bold fs-5'/>
            </Col>
            <Col/>
        </Form.Group>
        <Form.Group className="mb-2 d-flex justify-content-center">
            <Button variant="primary" size="lg" className='w-100 fw-bold text-light fs-3 compare-button' type="submit"
                disabled={isLoading || !couldCompare} 
                onClick={e => {
                    e.preventDefault(); 
                    Model.isQuestionEmpty() ? Model.compare_obj_asp() : Model.compare_question(); 
                    }}>
                {isLoading ? 'Comparing...' : 'Compare'}
            </Button>
        </Form.Group>
    </Form>);
}
