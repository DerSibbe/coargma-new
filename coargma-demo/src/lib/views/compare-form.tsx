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

    return (<Form>
        <Form.Group className="mb-3">
            <FloatingLabel label="Enter your question here...">
            <Form.Control type="text" placeholder="Enter your question here..."
                value={question} onChange={e => Model.setQuestion(e.target.value)} />
            </FloatingLabel>
        </Form.Group>
        
        <Form.Group className="mb-3">
            <Form.Text>or enter two objects for comparison</Form.Text>
        </Form.Group>

        <Form.Group as={Row} className="mb-3">
            <Col>
                <FloatingLabel label="Object 1">
                    <Form.Control type="text" placeholder="Object 1"
                        value={object1} onChange={e => Model.setObject1(e.target.value)} />
                </FloatingLabel>
            </Col>
            <Col>
                <FloatingLabel label="Object 2">
                    <Form.Control type="text" placeholder="Object 2"
                        value={object2} onChange={e => Model.setObject2(e.target.value)} />
                </FloatingLabel>
            </Col>
        </Form.Group>


        <Form.Group as={Row} className="mb-3">
            <Col>
                <FloatingLabel label="Aspect">
                    <Form.Control type="text" placeholder="Aspect"
                        value={aspect} onChange={e => Model.setAspect(e.target.value)} />
                </FloatingLabel>
            </Col>
            <Col></Col>
        </Form.Group>

        <Form.Group as={Row} className="mb-3 align-items-center">
            <Col>
                <Form.Label>Max number of arguments</Form.Label>
            </Col>
            <Col>
                <Form.Control type="number"
                    value={numberOfArguments} onChange={e => Model.setNumberOfArguments(+e.target.value)} />
            </Col>
            <Col>
                <Form.Check 
                    type="checkbox"
                    defaultChecked={true}
                    onChange={e => Model.setBaselineUse(e.target.checked)}
                    label="Use Baseline"
                />
            </Col>
            <Col />
        </Form.Group>


        <Form.Group className="mb-3 d-flex justify-content-center">
            <Button variant="primary" size="lg" type="submit"
                disabled={!couldCompare} onClick={e => { e.preventDefault(); Model.compare(); }}>
                Compare
            </Button>
        </Form.Group>
    </Form>);
}
