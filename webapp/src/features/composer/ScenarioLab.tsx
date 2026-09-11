import { useState, useMemo } from 'react'
import {
  BrainCircuit,
  ChartNoAxesCombined,
  Check,
  CheckCircle2,
  Database,
  Info,
  Network,
  SlidersHorizontal,
  Sparkles,
  ArrowRight,
  Shield,
  Layers,
  FlaskConical
} from 'lucide-react'
import type { ScenarioItem } from '../../contracts/types'

const domainIcons: Record<string, any> = {
  predictive_ml: BrainCircuit,
  deep_learning: Network,
  quantitative_finance: ChartNoAxesCombined,
  treasury: SlidersHorizontal,
}

export function ScenarioLab({
  scenarios,
  selectedScenarioId,
  onSelectScenario,
}: {
  scenarios: ScenarioItem[]
  selectedScenarioId: string | null
  onSelectScenario: (scenario: ScenarioItem) => void
}) {
  const [filterDomain, setFilterDomain] = useState<string>('all')

  const domains = [
    { id: 'all', label: 'All Scenarios' },
    { id: 'predictive_ml', label: 'Predictive ML' },
    { id: 'deep_learning', label: 'Deep Learning' },
    { id: 'quantitative_finance', label: 'Market / Quant' },
    { id: 'treasury', label: 'Treasury' },
  ]

  const filteredScenarios = useMemo(() => {
    if (filterDomain === 'all') return scenarios
    return scenarios.filter(s => s.domain === filterDomain)
  }, [scenarios, filterDomain])

  const activeScenario = useMemo(
    () => scenarios.find(s => s.id === selectedScenarioId) || null,
    [scenarios, selectedScenarioId]
  )

  return (
    <div className="scenario-lab-container" id="scenario-lab">
      {/* Scenario Lab Sub-header */}
      <div className="scenario-lab-header">
        <div>
          <h3 className="scenario-lab-title">Scenario Lab</h3>
          <p className="scenario-lab-desc">
            Select a verified built-in dataset or world generator to immediately profile and evaluate.
          </p>
        </div>

        {/* Domain Filter Pills */}
        <div className="scenario-filter-pills" role="tablist" aria-label="Scenario Domains">
          {domains.map(d => (
            <button
              key={d.id}
              role="tab"
              aria-selected={filterDomain === d.id}
              className={`filter-pill ${filterDomain === d.id ? 'active' : ''}`}
              onClick={() => setFilterDomain(d.id)}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {/* Active Selection Banner */}
      {activeScenario && (
        <div className="scenario-active-banner">
          <div className="active-banner-left">
            <CheckCircle2 size={16} className="text-sage" />
            <div>
              <strong>Active Data Scenario: {activeScenario.label}</strong>
              <span>
                {activeScenario.shape} · target: {activeScenario.target} · mapped context: <code>{activeScenario.compatible_context_id}</code>
              </span>
            </div>
          </div>
          <span className="banner-applicability-tag">
            {activeScenario.applicable_tests_count} applicable tests
          </span>
        </div>
      )}

      {/* Scenario Cards Grid */}
      <div className="scenario-cards-grid">
        {filteredScenarios.map(sc => {
          const isSelected = selectedScenarioId === sc.id
          const I = domainIcons[sc.domain] || Database

          const classificationBadge = sc.classification === 'CANONICAL_EXECUTION_CONTEXT'
            ? 'CANONICAL CONTEXT'
            : sc.classification === 'GENERATOR_ONLY'
            ? 'GENERATOR ONLY'
            : 'DATA SCENARIO'

          const classificationClass = sc.classification === 'CANONICAL_EXECUTION_CONTEXT'
            ? 'pill-sage'
            : sc.classification === 'GENERATOR_ONLY'
            ? 'pill-amber'
            : 'pill-accent'

          return (
            <div
              key={sc.id}
              id={`scenario-card-${sc.id}`}
              className={`scenario-card ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectScenario(sc)}
              style={{ cursor: 'pointer' }}
            >
              {/* Card Top: Icon, Domain, Classification */}
              <div className="scenario-card-top">
                <div className="scenario-card-icon">
                  <I size={15} />
                </div>
                <span className="scenario-domain-text">
                  {sc.domain === 'predictive_ml'
                    ? 'Predictive ML'
                    : sc.domain === 'deep_learning'
                    ? 'Deep Learning'
                    : sc.domain === 'quantitative_finance'
                    ? 'Market / Quant'
                    : 'Treasury'}
                </span>
                <span className={`status-pill ${classificationClass}`}>
                  {classificationBadge}
                </span>
              </div>

              {/* Title & Description */}
              <h4 className="scenario-card-title">{sc.label}</h4>
              <p className="scenario-card-desc">{sc.description}</p>

              {/* Volumetrics & Target */}
              <div className="scenario-specs-row">
                <span>{sc.shape}</span>
                <span>·</span>
                <span>Target: <b>{sc.target}</b></span>
              </div>

              {/* Categories */}
              <div className="scenario-tags-row">
                {sc.categories.map(cat => (
                  <span key={cat} className="scenario-tag">{cat}</span>
                ))}
              </div>

              {/* Test Counts & Context Mapping */}
              <div className="scenario-mapping-box">
                <div className="mapping-tests-count">
                  <FlaskConical size={12} className="text-muted" />
                  <span><b>{sc.registered_tests_count}</b> registered · <b>{sc.applicable_tests_count}</b> applicable</span>
                </div>
                <div className="mapping-context-link">
                  <span>Context: <code>{sc.compatible_context_id}</code></span>
                </div>
              </div>

              {/* Card Action Button */}
              <div className="scenario-card-actions">
                <button
                  id={`use-scenario-${sc.id}`}
                  className={`scenario-select-btn ${isSelected ? 'active' : ''}`}
                  onClick={() => onSelectScenario(sc)}
                >
                  {isSelected ? (
                    <>
                      <Check size={14} /> Active scenario
                    </>
                  ) : (
                    <>
                      Use scenario <ArrowRight size={13} />
                    </>
                  )}
                </button>
              </div>

              {/* Subtle Provenance Hover Footer */}
              <div className="scenario-card-hover-info">
                <Info size={11} />
                <span title={sc.provenance_note}>{sc.generator_identity} (seed {sc.seed})</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
