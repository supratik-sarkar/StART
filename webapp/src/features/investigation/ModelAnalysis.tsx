import { SensitivityResults, TuningResults } from './ExperimentResults'
import type { ArtifactRecord, RuntimeEvent } from '../../contracts/types'
import { ConfigurationCodeView } from '../canvas/ConfigurationCodeView'
import { Bars, ConfusionMatrix, DataBlock, DataValue, Empty, LineFigure, Matrix, ScoreStrip, Section, ScientificTable } from './Science'
import { label, object, type ScientificRecord } from './presentation'

export function analysisSections(family: string) {
  if (family === 'deep_learning') return ['Overview','Data','Architecture','Training','Performance','Explainability','Sensitivity','Tuning','Validation']
  if (family === 'recommender') return ['Overview','Interaction Data','Model','Evaluation Protocol','Ranking / Rating','Coverage & Cold Start','Sensitivity','Validation']
  if (family === 'portfolio' || family === 'quantitative_finance') return ['Overview','Universe & Data','Construction','Allocation','Risk','Performance','Sensitivity','Stability']
  return ['Overview','Data','Model','Performance','Explainability','Sensitivity','Tuning','Validation']
}
export function ArtifactFigure({ artifact }: { artifact: ArtifactRecord }) {
  const raw = artifact.content ?? artifact.preview?.payload
  const svg = typeof raw === 'string' && raw.trim().startsWith('<svg') ? raw : raw?.dendrogram_svg ?? raw?.svg
  if (svg) return <figure className="source-figure"><img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`} alt={artifact.label}/><figcaption>{artifact.description || artifact.label} · supplied artifact</figcaption></figure>
  if (raw?.url && /^(https?:|\/)/.test(raw.url) && !raw.url.startsWith('//')) return artifact.mimeType === 'application/pdf' ? <iframe className="source-pdf" src={raw.url} title={artifact.label}/> : <img className="source-image" src={raw.url} alt={artifact.label}/>
  return <DataValue value={raw}/>
}
export function Performance({ a }: { a: ScientificRecord }) {
  const d = object(a.diagnostics), roc = object(d.roc_curve), cal = object(d.calibration), pr = object(d.pr_curve ?? d.precision_recall_curve)
  const portfolio = ['portfolio','quantitative_finance'].includes(a.model_family)
  return <>
    <Section title="Metric summary" note="Canonical model result · exact values remain inspectable"><ScoreStrip data={object(a.metrics)}/><details><summary>Inspect exact metrics</summary><DataValue value={a.metrics}/></details></Section>
    {!portfolio && <><div className="figure-grid"><Section title="Discrimination" note="Receiver operating characteristic"><LineFigure unitSquare reference xLabel="False positive rate" yLabel="True positive rate" series={[{name:'ROC', x:roc.fpr ?? [],y:roc.tpr ?? []}]}/></Section><Section title="Confusion matrix" note="Holdout observations · counts"><ConfusionMatrix data={object(d.confusion_matrix)}/></Section></div>
    <Section title="Calibration" note="Reliability diagnostic · predicted probability against observed frequency"><ScoreStrip data={{...(cal.ece != null ? {ece:cal.ece} : {}), ...(a.metrics?.brier_score != null ? {brier_score:a.metrics.brier_score} : {})}}/><LineFigure unitSquare reference xLabel="Predicted probability" yLabel="Observed frequency" series={[{name:'Observed reliability',x:cal.predicted_probabilities ?? cal.mean_predicted_value ?? [],y:cal.observed_probabilities ?? cal.fraction_of_positives ?? []}]}/></Section>
    <Section title="Precision–recall"><LineFigure unitSquare xLabel="Recall" yLabel="Precision" series={[{name:'Precision–recall',x:pr.recall ?? [],y:pr.precision ?? []}]}/></Section><DataBlock title="Threshold analysis" data={d.threshold_analysis}/><DataBlock title="Lift / decile" data={d.lift ?? d.decile_analysis}/></>}
    {portfolio && <DataBlock title="Performance history" data={d.performance_history ?? d.cumulative_returns}/>}
  </>
}
export function ModelAnalysis({ section, a, artifacts, events=[] }: { section: string; a: ScientificRecord; artifacts: ArtifactRecord[]; events?:RuntimeEvent[] }) {
  const d = object(a.diagnostics), s = object(a.structural_analysis), config = object(a.resolved_configuration)
  const family = a.model_family, hrp = ['hierarchical_risk_parity','hrp'].includes(a.technique)
  if (section === 'Data' || section === 'Interaction Data' || section === 'Universe & Data') return <>
    <DataBlock title={section === 'Interaction Data' ? 'Interaction population' : section === 'Universe & Data' ? 'Investment universe' : 'Dataset'} data={a.data_selection ?? (config.assets ? {assets: config.assets} : undefined)}/>
    <DataBlock title="Integrity & missingness" data={a.data_validation}/><DataBlock title="Feature roles" data={a.data_selection?.semantic_roles}/>
    <DataBlock title={family === 'recommender' ? 'Interaction distribution & popularity' : 'Preprocessing'} data={family === 'recommender' ? d.interaction_distribution ?? d.popularity_distribution : a.preprocessing}/>
    {s.correlation_matrix && <Section title="Correlation"><Matrix data={s.correlation_matrix}/></Section>}
    {section === 'Universe & Data' && <DataBlock title="Returns & covariance" data={s.covariance_matrix ?? d.covariance ?? a.returns}/>}
  </>
  if (section === 'Model') return <><DataBlock title="Evaluation design" data={a.split_protocol}/><Section title="Resolved configuration"><ConfigurationCodeView rawConfig={a.resolved_configuration}/></Section><DataBlock title="Fitted model summary" data={a.execution_summary ?? d.model_summary}/></>
  if (section === 'Performance') return <Performance a={a}/>
  if (section === 'Sensitivity') return <SensitivityResults data={a.sensitivity}/>
  if (section === 'Tuning') return <TuningResults data={a.tuning ?? d.tuning} events={events}/>
  if (section === 'Explainability') return <>
    <Section title={s.permutation_importance ? 'Permutation importance' : 'Model explainability'} note={s.permutation_importance ? 'Ranked by supplied importance · bar length represents magnitude' : undefined}>{s.permutation_importance ? <Bars data={s.permutation_importance} name="Permutation importance"/> : Object.keys(s).length ? <DataValue value={s}/> : <Empty title="Explainability not supplied"/>}</Section>
    {Object.entries(s).filter(([key])=>['native_importance','tree_shap','kernel_shap','shap','pdp','ice','ale','lime'].includes(key)).map(([key,value])=><DataBlock key={key} title={key==='native_importance'?'Native importance':key.toUpperCase().replace(/_/g,' ')} data={value}/>)}
    <DataBlock title="Method & limitations" data={s.method ?? s.limitations}/>
  </>
  if (section === 'Validation' || section === 'Stability') return <>
    <DataBlock title="Sensitivity & perturbation" data={a.sensitivity}/><DataBlock title="Robustness" data={d.robustness ?? a.robustness}/><DataBlock title="Stability" data={d.stability ?? a.stability}/><DataBlock title="Baseline / challenger" data={a.baseline_comparison ?? d.baseline_comparison}/>
  </>
  if (section === 'Architecture') {
    const arch = object(d.architecture ?? s.architecture)
    return <><Section title="Network architecture" note={config.architecture ? `${config.architecture} · supplied layer order` : 'Supplied layer order'}>{Array.isArray(arch.layers) ? <ol className="network-layers">{arch.layers.map((layer: unknown,i: number) => <li key={i}><span>{String(i+1).padStart(2,'0')}</span><div><DataValue value={layer}/></div></li>)}</ol> : <Empty title="Layer hierarchy not supplied"/>}</Section><DataBlock title="Parameters & tensor shapes" data={Object.fromEntries(Object.entries(arch).filter(([k]) => k !== 'layers'))}/><Section title="Resolved configuration"><ConfigurationCodeView rawConfig={a.resolved_configuration}/></Section></>
  }
  if (section === 'Training') {
    const c = object(d.training_curve)
    return <><DataBlock title="Training specification" data={{optimizer:config.optimizer, loss:config.loss, learning_rate:config.learning_rate, batch_size:config.batch_size, epochs:config.epochs, device:d.architecture?.device ?? config.device}}/><Section title="Convergence" note="Observed training and validation histories"><LineFigure xLabel="Epoch" yLabel="Loss" series={[{name:'Training',x:c.epochs ?? [],y:c.train_loss ?? []},{name:'Validation',x:c.epochs ?? [],y:c.val_loss ?? []}]}/></Section><DataBlock title="Learning-rate trace" data={d.learning_rate_trace ?? c.learning_rates}/><DataBlock title="Gradient diagnostics" data={d.gradients ?? d.gradient_norms}/><DataBlock title="Checkpoint selection" data={d.checkpoints}/></>
  }
  if (section === 'Evaluation Protocol') return <DataBlock title="Evaluation protocol" data={{feedback:a.data_selection?.feedback, ...object(a.split_protocol), candidate_universe:a.split_protocol?.candidate_universe, seen_item_exclusion:a.split_protocol?.seen_item_exclusion, negative_sampling:a.split_protocol?.negative_sampling, configured_k:config.k_cutoffs ?? d.ranking?.k_list, cold_start_protocol:a.split_protocol?.cold_start_protocol}}/>
  if (section === 'Ranking / Rating') {
    const r = object(d.ranking)
    const rows = (r.k_list ?? []).map((k: number) => ({K:k, Precision:r.precision_at_k?.[k],Recall:r.recall_at_k?.[k],NDCG:r.ndcg_at_k?.[k],HitRate:r.hit_rate_at_k?.[k],MAP:r.map_at_k?.[k]}))
    return <><Section title="Ranking quality" note="Metrics at supplied cutoffs"><ScientificTable rows={rows}/><DataValue value={{mrr:r.mrr,users_evaluated:r.n_users_evaluated}}/></Section><Section title="Rating quality">{a.metrics?.rmse != null || a.metrics?.mae != null ? <ScoreStrip data={{rmse:a.metrics?.rmse,mae:a.metrics?.mae,r2:a.metrics?.r2}}/> : <Empty title={['top_k_ranking','contextual_ranking'].includes(a.task_type) ? 'Not applicable' : 'Rating metrics not supplied'}>This run supplies no rating error metrics.</Empty>}</Section></>
  }
  if (section === 'Coverage & Cold Start') return <><DataBlock title="Coverage & beyond accuracy" data={d.beyond_accuracy}/><DataBlock title="Cold-start cohorts" data={d.cold_start} note="Cohort counts and values are reported as supplied; empty cohorts do not establish model performance."/></>
  if (section === 'Construction') return <>
    {hrp ? <><Section title="Hierarchical risk parity" note="Construction sequence · each output is inspected from its source"><ol className="construction-chain">{['Returns','Correlation','Distance','Hierarchical clustering','Linkage','Dendrogram','Quasi-diagonal order','Recursive bisection','Weights','Risk contribution'].map(t => <li key={t}>{t}</li>)}</ol></Section><div className="figure-grid"><Section title="Correlation"><Matrix data={s.correlation_matrix}/></Section><Section title="Distance"><Matrix data={s.distance_matrix}/></Section></div><Section title="Dendrogram" note="Original supplied figure">{artifacts.find(a => /DENDROGRAM/i.test(a.artifactId)) ? <ArtifactFigure artifact={artifacts.find(a => /DENDROGRAM/i.test(a.artifactId))!}/> : <Empty title="Dendrogram figure not supplied">The linkage remains inspectable below. No tree is inferred.</Empty>}</Section><Section title="Linkage" note="Source node indices, distance, and cluster size">{Array.isArray(s.linkage?.linkage_matrix) ? <ScientificTable rows={s.linkage.linkage_matrix.map((row: number[]) => ({left_node:row[0],right_node:row[1],distance:row[2],cluster_size:row[3]}))}/> : <Empty/>}</Section><DataBlock title="Quasi-diagonal order" data={s.dendrogram?.quasi_diagonal_order ?? s.quasi_diagonal_order}/><DataBlock title="Clusters & recursive bisection" data={s.clusters ?? s.recursive_bisection}/></> : <><Section title={a.technique === 'equal_risk_contribution' ? 'Risk budgeting' : 'Constrained optimization'}><ConfigurationCodeView rawConfig={a.resolved_configuration}/></Section><DataBlock title="Covariance" data={s.covariance_matrix ?? d.covariance}/><DataBlock title="Solver diagnostics" data={a.execution_summary ?? {converged:a.metrics?.converged}}/></>}
  </>
  if (section === 'Allocation') return <><Section title="Portfolio weights" note="Exact canonical allocations"><Bars data={object(d.weights)} name="Portfolio weights"/></Section><DataBlock title="Concentration & effective holdings" data={{herfindahl_index:a.metrics?.herfindahl_index,effective_n_positions:a.metrics?.effective_n_positions, ...object(d.concentration)}}/>{hrp && <DataBlock title="Cluster allocation" data={s.cluster_allocation ?? d.cluster_allocation}/>}</>
  if (section === 'Risk') return <><Section title="Component risk contributions"><Bars data={object(d.risk_contributions?.component_risk_contributions)}/></Section><Section title="Risk contribution fractions"><Bars data={object(d.risk_contributions?.percentage_risk_contributions)}/></Section><DataBlock title="Risk budgets & parity" data={{risk_budget:config.risk_budget,target_risk_contribution:a.metrics?.target_risk_contribution,parity_deviation:a.metrics?.max_risk_contribution_dispersion, ...object(d.risk_budgets)}}/><DataBlock title="Marginal contributions" data={d.risk_contributions?.marginal_risk_contributions}/></>
  return <Empty title={`${label(section)} not supplied`}/>
}
