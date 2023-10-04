import Model from '../../lib/model';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const summary = useObservable(() => Model.summary) || [];

    return (<>
        {summary.length > 0 ? <>
            <br/>
        </> : <></>}
    </>);
}
