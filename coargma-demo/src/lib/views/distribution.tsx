import ProgressBar from "react-bootstrap/ProgressBar";
import { useObservable } from "rxjs-hooks";
import Model from "../model";

export default function() {
    const distribution = useObservable(() => Model.distribution);

    return (
        <>
          {distribution ? (
            <>
              <br />
              <div style={{ position: 'relative', width: '100%', height: '50px', backgroundColor: '#A6241D' , borderRadius: '5px'}}>
                <div
                  style={{
                    height: '100%',
                    width: `${distribution?.object1}%`,
                    backgroundColor: '#168124', borderRadius: '5px'
                  }}
                />
                <div style={{ position: 'absolute', top: '50%', left: `${distribution?.object1 / 2}%`, transform: 'translateY(-50%)' , borderRadius: '5px'}}>
                  {`${distribution?.object1}%`}
                </div>
                <div style={{ position: 'absolute', top: '50%', left: `${distribution?.object1 + (distribution?.object2 / 2)}%`, transform: 'translateY(-50%)' , borderRadius: '5px'}}>
                  {`${distribution?.object2}%`}
                </div>
              </div>
            </>
          ) : (
            <></>
          )}
        </>
      );
      
      
}
