import ProgressBar from "react-bootstrap/ProgressBar";
import { useObservable } from "rxjs-hooks";
import Model from "../model";

export default function() {
    const distribution = useObservable(() => Model.distribution);

    return (<>
        {distribution ? <>
            <br/>
            <ProgressBar style={{height: '30px'}}>
                <ProgressBar variant="success"
                    label={`${distribution?.object1}%`}  now={distribution?.object1}
                    key={1} />
                <ProgressBar
                    label={`${distribution?.object2}%`} now={distribution?.object2}
                    key={2} />
            </ProgressBar>
        </> : <></>}
    </>);
}
