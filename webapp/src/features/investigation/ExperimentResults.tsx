import type { RuntimeEvent } from '../../contracts/types'
import { DataBlock, DataValue, Empty, LineFigure, ScientificTable, Section } from './Science'
import { object, type ScientificRecord } from './presentation'

// Re-arrange supplied rows for display; never calculate metrics or missing shocks.
export function sensitivitySeries(data: ScientificRecord) {
  const rows:ScientificRecord[]=Array.isArray(data.rows)?data.rows:[]
  const features=[...new Set(rows.map(r=>r.feature).filter(v=>typeof v==='string'))]
  return features.map(name=>{const points=rows.filter(r=>r.feature===name&&typeof r.shock==='number'&&typeof r.metric==='number');return {name,x:points.map(r=>r.shock),y:points.map(r=>r.metric)}})
}
export function SensitivityResults({data}:{data:unknown}) {
  const d=object(data), series=sensitivitySeries(d)
  return <><Section title="Sensitivity analysis" note="Backend-produced observations; no inferred shocks or metric deltas"><DataValue value={{metric:d.metric_name,baseline:d.baseline,mode:d.mode,retraining:d.retraining}}/>{series.length>0?<LineFigure xLabel="Shock (fraction)" yLabel={d.metric_name??'Supplied metric'} series={series}/>:<Empty title="Shock response curves not supplied"/>}</Section>{Array.isArray(d.rows)&&<Section title="Exact shock observations"><ScientificTable rows={d.rows}/></Section>}{d.drift_table&&<Section title="Sensitivity matrix · supplied drift"><ScientificTable rows={Object.entries(object(d.drift_table)).map(([feature,values])=>({feature,...object(values)}))}/></Section>}<DataBlock title="Available sensitivity outputs" data={data}/></>
}
export function TuningResults({data,events=[]}:{data:unknown;events?:RuntimeEvent[]}) {
 const supplied=object(data), summary=events.find(e=>e.metadata?.best_hyperparameters)?.metadata??{}, d={...summary,...supplied}, rows=Array.isArray(d.trials)?d.trials:events.filter(e=>String(e.type)==='tuning_trial'&&typeof e.metadata?.trial==='number').map(e=>({...e.metadata,status:e.status,event_id:e.eventId}))
 return <><DataBlock title="Search outcome" data={Object.keys(d).length?{objective:d.metric_name??d.objective_metric,model:d.model,strategy:d.strategy??d.tuning_strategy,best_hyperparameters:d.best_params??d.best_hyperparameters,best_metric:d.best_metric}:undefined}/><Section title="Trial table"><ScientificTable rows={rows}/></Section><Section title="Optimization history" note="Supplied trial observations"><LineFigure xLabel="Trial" yLabel={d.metric_name??d.objective_metric??rows[0]?.objective_metric??'Validation metric (backend name not supplied)'} series={[{name:'Validation metric',x:rows.filter((r:ScientificRecord)=>typeof r.trial==='number'&&typeof r.validation_metric==='number').map((r:ScientificRecord)=>r.trial),y:rows.filter((r:ScientificRecord)=>typeof r.trial==='number'&&typeof r.validation_metric==='number').map((r:ScientificRecord)=>r.validation_metric)}]}/></Section><DataBlock title="Parameter importance" data={d.parameter_importance}/><DataBlock title="Champion versus baseline" data={d.baseline_comparison}/></>
}
