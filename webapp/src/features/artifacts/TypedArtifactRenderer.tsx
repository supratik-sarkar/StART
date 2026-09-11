import type { ArtifactRecord } from '../../contracts/types'
import { ArtifactFigure } from '../investigation/ModelAnalysis'
import { Bars, ConfusionMatrix, LineFigure } from '../investigation/Science'
import { object } from '../investigation/presentation'
export function TypedArtifactRenderer({artifact}: {artifact:ArtifactRecord;onHighlightEvidence?:(id:string)=>void}) {
  const c=object(artifact.content ?? artifact.preview?.payload), id=artifact.artifactId
  if(/CONFUSION/i.test(id))return <ConfusionMatrix data={c}/>
  if(/ROC/i.test(id))return <LineFigure unitSquare reference xLabel="False positive rate" yLabel="True positive rate" series={[{name:'ROC',x:c.fpr ?? [],y:c.tpr ?? []}]}/>
  if(/CALIBRATION/i.test(id))return <LineFigure unitSquare reference xLabel="Predicted probability" yLabel="Observed frequency" series={[{name:'Reliability',x:c.predicted_probabilities ?? [],y:c.observed_probabilities ?? []}]}/>
  if(/IMPORTANCE/i.test(id))return <Bars data={c.permutation_importance ?? c.importance_scores ?? {}} name="Supplied feature importance"/>
  if(/TRAINING/i.test(id))return <LineFigure xLabel="Epoch" yLabel="Loss" series={[{name:'Training',x:c.epochs ?? [],y:c.train_loss ?? []},{name:'Validation',x:c.epochs ?? [],y:c.val_loss ?? []}]}/>
  return <ArtifactFigure artifact={artifact}/>
}
