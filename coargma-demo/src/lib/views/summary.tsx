import Model from '../../lib/model';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const summary = useObservable(() => Model.summary) || [];

    return (<>
        {summary.length > 0 ? <>
            <br/>
            <h2>Summary</h2>
            <div style={{background: "lightgrey"}}>
            {summary.map(line => (<p>
                &nbsp;{line}
            </p>))}
            </div>
        </> : <></>}
    </>);
}
