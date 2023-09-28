import Model from '../../lib/model';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const sources = useObservable(() => Model.sources) || [];
    
    return (<>
        {sources.length > 0 ? <>
            <br/>
            <h2>Sources</h2>
            <ol type="1">
            {sources.map(source => (<li>
                <a href={source.url}>{source.caption}</a>
            </li>))}
            </ol>
        </> : <></>}
    </>);
}