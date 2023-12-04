import Model from '../../lib/model';
import { useObservable } from 'rxjs-hooks';

export default function() {
    const sources_obj1 = useObservable(() => Model.sources_obj1) || [];
    const sources_obj2 = useObservable(() => Model.sources_obj2) || [];
    
    return (<>
        {sources_obj1.length > 0 ? <>
            <br/>
            
 <table>
  <tbody>
    <tr>
      <td style={{verticalAlign: "top"}}>
        <h2>For object 1</h2>
        <ol type="1">
            {sources_obj1.map(source => (<li>
                <a href={source.url}>{source.caption}</a>
            </li>))}
        </ol>
      </td>
      <td style={{verticalAlign: "top"}}>
        <h2>For object 2</h2>
        <ol type="1">
            {sources_obj2.map(source => (<li>
                <a href={source.url}>{source.caption}</a>
            </li>))}
        </ol>
      </td>
    </tr>
  </tbody>
</table>
        </> : <></>}
    </>);
}