import Model from '../../lib/model';
import Form from 'react-bootstrap/Form';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const sources_obj1 = useObservable(() => Model.sources_obj1) || [];
    const sources_obj2 = useObservable(() => Model.sources_obj2) || [];
    const object1 = useObservable(() => Model.compare_object1) || "";
    const object2 = useObservable(() => Model.compare_object2) || "";
    
    return (<>
        {sources_obj1.length > 0 ? <>
            <br/>
            
 <table>
  <tbody>
    <tr>
      <td className='sources-first-object'>
      <h2 className='h2-compare-tittle-object1'>{object1}</h2>
        <ol type="1">
            {sources_obj1.map(source => (<li className='list-margins'>
                <a href={source.url} className='sources-list' target="_blank" rel="noopener noreferrer">{source.caption}</a>
            </li>))}
        </ol>
      </td>
      <td className='sources-second-object'>
        <h2 className='h2-compare-tittle-object2'>{object2}</h2>
        <ol type="1">
            {sources_obj2.map(source => (<li className='list-margins'>
                <a href={source.url} className='sources-list' target="_blank" rel="noopener noreferrer">{source.caption}</a>
            </li>))}
        </ol>
      </td>
    </tr>
  </tbody>
</table>
        </> : <></>}
    </>);
}