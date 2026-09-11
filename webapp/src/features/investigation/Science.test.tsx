import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ConfusionMatrix, LineFigure } from './Science'
import { ConfigurationCodeView, pythonLiteral } from '../canvas/ConfigurationCodeView'
import { analysisSections, ModelAnalysis } from './ModelAnalysis'
import { ScientificComparison } from '../compare/RunCompareView'
import { isAnalyticalObject } from './presentation'
import type { ArtifactRecord } from '../../contracts/types'

const render = renderToStaticMarkup

describe('Scientific presentation truth', () => {
  it('does not derive a confusion matrix from accuracy or precision', () => {
    const html = render(<ConfusionMatrix data={{accuracy:.9,precision:.8}}/>)
    expect(html).toContain('Confusion matrix not supplied')
    expect(html).not.toContain('<table')
  })
  it('renders supplied confusion counts including zero', () => {
    const html = render(<ConfusionMatrix data={{tn:14,fp:0,fn:3,tp:8}}/>)
    expect(html).toContain('<strong>0</strong>')
    expect(html).toContain('<strong>14</strong>')
    expect(html).not.toContain('NaN')
  })
  it('refuses to manufacture or pair incomplete curve observations', () => {
    const html = render(<LineFigure xLabel="FPR" yLabel="TPR" series={[{name:'ROC',x:[0,.5,1],y:[0,1]}]}/>)
    expect(html).toContain('Curve not supplied')
    expect(html).not.toContain('<svg')
  })
  it('preserves nested configuration and Python scalar syntax', () => {
    expect(pythonLiteral({seed:0,enabled:false,optional:null,nested:{values:[0,true]}})).toContain('"values": [0, True]')
    expect(pythonLiteral({enabled:false,optional:null})).toContain('"enabled": False')
    expect(pythonLiteral({enabled:false,optional:null})).toContain('"optional": None')
    expect(render(<ConfigurationCodeView workflowId="predictive_ml" contextId="context"/>)).toContain('Resolved configuration not supplied')
  })
  it('keeps raw metadata out of the analytical pane', () => {
    const a = {artifactId:'metadata',kind:'json',artifactType:'resolved_configuration',mimeType:'application/json',content:{seed:42}} as ArtifactRecord
    expect(isAnalyticalObject(a)).toBe(false)
    expect(isAnalyticalObject({...a,kind:'plot',artifactType:'roc_curve',content:null})).toBe(false)
    expect(isAnalyticalObject({...a,kind:'plot',artifactType:'roc_curve',content:{fpr:[0,1],tpr:[0,1]}})).toBe(true)
  })
  it('shows backend incompatibility without any metric deltas', () => {
    const html = render(<ScientificComparison result={{compatible:false,incompatibleReason:'Different candidate universes',parameters:[{param:'secret_delta',valA:1,valB:2,changed:true}]}}/>)
    expect(html).toContain('Different candidate universes')
    expect(html).not.toContain('secret_delta')
  })
  it('does not apply HRP hierarchy to minimum variance', () => {
    const html = render(<ModelAnalysis section="Construction" a={{model_family:'portfolio',technique:'minimum_variance',resolved_configuration:{solver:'SLSQP'}}} artifacts={[]}/>)
    expect(html).toContain('Constrained optimization')
    expect(html).not.toContain('Dendrogram')
  })
  it('has family-specific investigation sections without an Agent section', () => {
    expect(analysisSections('deep_learning')).toContain('Architecture')
    expect(analysisSections('recommender')).toContain('Evaluation Protocol')
    expect(analysisSections('portfolio')).toContain('Allocation')
    expect(analysisSections('predictive_ml')).not.toContain('Agent')
  })
})
