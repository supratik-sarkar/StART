import type { ExecutionContext, RunRequest, ScenarioItem, WorkflowId } from '../../contracts/types'

// Source-backed transport projection; scientific computation stays in the backend.
export function runConfiguration(workflow: WorkflowId | null, seed?: number, trials?: number, parameters: Record<string,any> = {}, executionMode: import('../../contracts/types').ExecutionMode = 'hybrid_workbench'): Pick<RunRequest, 'parameters' | 'seed' | 'executionMode'> {
  if (seed !== undefined && (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295)) throw new Error('Seed must be an integer between 0 and 4294967295.')
  if (workflow === 'hyperparameter_tuning' && trials !== undefined && (!Number.isInteger(trials) || trials < 5 || trials > 30)) throw new Error('Trial budget must be an integer between 5 and 30.')
  if(parameters.split?.test_size!==undefined && (parameters.split.test_size<=0 || parameters.split.test_size>=1)) throw new Error('Test fraction must be between zero and one.')
  const clean=JSON.parse(JSON.stringify(parameters,(_k,v)=>v===''?undefined:v))
  return { executionMode, ...(seed === undefined ? {} : {seed}), parameters:{...clean,...(workflow==='hyperparameter_tuning'&&trials!==undefined?{trials}:{})} }
}
export function compatibleContexts(contexts: ExecutionContext[], scenarios: ScenarioItem[], workflow: WorkflowId | null) {
  if (!workflow) return contexts
  if(workflow==='fraud_anomaly_aml')return contexts.filter(c=>c.id==='synthetic_aml_imbalanced')
  if(workflow==='recommender_system')return contexts.filter(c=>c.id.startsWith('recommender_'))
  return contexts.filter(c => scenarios.some(s => s.id === c.id && s.compatible_context_id === c.id && s.compatible_workflows.includes(workflow)))
}
export const capabilityLabel = (key: string) => ({auc_roc:'ROC-AUC',pr_auc:'PR-AUC',mlp:'Tabular MLP',lightgbm:'LightGBM',xgboost:'XGBoost',catboost:'CatBoost',logistic_regression:'Logistic Regression · baseline candidate',matrix_factorization:'Matrix Factorization',neural_collaborative_filtering:'Neural Collaborative Filtering',factorization_machine:'Factorization Machine',field_aware_factorization_machine:'Field-Aware Factorization Machine',tree_shap:'Tree SHAP',kernel_shap:'Kernel SHAP',optuna_bayesian:'Bayesian / Optuna',hierarchical_risk_parity:'HRP',minimum_variance:'Minimum Variance',equal_risk_contribution:'ERC / Risk Parity',lstm:'LSTM',gru:'GRU',rnn:'RNN',bi_lstm:'Bidirectional LSTM'}[key] ?? key.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()))
