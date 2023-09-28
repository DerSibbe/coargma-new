'use client';

import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import CompareForm from '../lib/views/compare-form';
import Summary from '../lib/views/summary';
import Sources from '../lib/views/sources';
import Distribution from '../lib/views/distribution';

export default function Home() {
  return (
    <Container>
      <Row>
        <Col><br/></Col>
      </Row>
      <Row>
        <Col>
          <h1>Comparative Argumentative Machine v2</h1>
        </Col>
      </Row>
      <Row>
        <Col><br/><br/></Col>
      </Row>
      <Row>
        <Col>
          <CompareForm />
        </Col>
      </Row>
      <Row>
        <Col>
          <Distribution />
        </Col>
      </Row>
      <Row>
        <Col>
          <Summary />
        </Col>
      </Row>
      <Row>
        <Col>
          <Sources />
        </Col>
      </Row>
    </Container>
  );
}
