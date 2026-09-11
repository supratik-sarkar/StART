import { useState, useMemo, useEffect } from 'react'
import {
  X,
  Search,
  BookOpen,
  FlaskConical,
  CheckCircle2,
  CircleSlash2,
  HelpCircle,
  Filter,
  Check,
  Tag
} from 'lucide-react'
import type { TestCatalogItem, ExecutionContext, WorkflowId } from '../../contracts/types'

export function TestCatalogDrawer({
  isOpen,
  onClose,
  tests,
  activeContextId,
  activeWorkflowId,
}: {
  isOpen: boolean
  onClose: () => void
  tests: TestCatalogItem[]
  activeContextId: string | null
  activeWorkflowId: WorkflowId | null
}) {
  const [searchQuery, setSearchQuery] = useState('')
  const [domainFilter, setDomainFilter] = useState<'all' | 'predictive_ml' | 'quantitative_finance' | 'treasury'>('all')
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null)

  // 1. Domain counts (Authoritative Mechanical Inventory)
  const totalCount = tests.length
  const predCount = tests.filter(t => t.domain === 'predictive_ml').length
  const marketCount = tests.filter(t => t.domain === 'quantitative_finance').length
  const treasuryCount = tests.filter(t => t.domain === 'treasury').length

  // 2. Filter tests
  const filteredTests = useMemo(() => {
    return tests.filter(t => {
      // Domain filter
      if (domainFilter !== 'all' && t.domain !== domainFilter) return false
      // Family filter
      if (selectedFamily && t.family !== selectedFamily) return false
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase()
        const matchesId = t.testId.toLowerCase().includes(q)
        const matchesName = t.name.toLowerCase().includes(q)
        const matchesDesc = t.description.toLowerCase().includes(q)
        const matchesFamily = t.family.toLowerCase().includes(q)
        if (!matchesId && !matchesName && !matchesDesc && !matchesFamily) return false
      }
      return true
    })
  }, [tests, domainFilter, selectedFamily, searchQuery])

  // Extract unique families for chips
  const availableFamilies = useMemo(() => {
    const list = domainFilter === 'all' ? tests : tests.filter(t => t.domain === domainFilter)
    return Array.from(new Set(list.map(t => t.family))).sort()
  }, [tests, domainFilter])

  // Determine test applicability against the active context / workflow
  const getApplicability = (t: TestCatalogItem) => {
    if (!activeContextId && !activeWorkflowId) {
      return { status: 'REGISTERED', reason: 'Select a scenario and workflow to evaluate applicability.' }
    }

    if (activeContextId === 'institutional_market_v1') {
      if (t.domain === 'quantitative_finance') {
        return { status: 'APPLICABLE', reason: 'Applicable to multi-asset market context.' }
      }
      if (t.domain === 'treasury') {
        return { status: 'APPLICABLE', reason: 'Short-rate series included in market context bundle.' }
      }
      return { status: 'NOT_APPLICABLE', reason: 'Requires tabular dataset context; active context is multi-asset market world.' }
    }

    if (activeContextId === 'deep_learning_v1') {
      if (t.domain === 'predictive_ml') {
        // DL candidate tests
        if (t.testId.startsWith('supervised.') || t.testId.startsWith('xai.')) {
          return { status: 'APPLICABLE', reason: 'Candidate diagnostic for deep learning neural fixture.' }
        }
        return { status: 'NOT_APPLICABLE', reason: 'Traditional tabular feature engineering test skipped for neural latent benchmark.' }
      }
      return { status: 'NOT_APPLICABLE', reason: 'Requires market risk world; active context is tabular deep learning.' }
    }

    // Default tabular (institutional_credit_v1, synthetic_aml_imbalanced, etc.)
    if (t.domain === 'predictive_ml') {
      return { status: 'APPLICABLE', reason: 'Candidate validation diagnostic for tabular credit risk benchmark.' }
    }
    return { status: 'NOT_APPLICABLE', reason: 'Requires multi-asset market world; active context is tabular credit risk.' }
  }

  // Escape key listener
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="test-catalog-overlay" onClick={onClose} data-testid="test-catalog-overlay">
      <aside className="test-catalog-drawer" onClick={e => e.stopPropagation()} role="dialog" aria-label="Test Catalog">
        {/* Drawer Header */}
        <header className="catalog-drawer-header">
          <div className="catalog-header-title">
            <BookOpen size={18} className="text-accent" />
            <div>
              <h3>Canonical {totalCount}-Test Catalog</h3>
              <p>Authoritative test registry from deterministic engineering engines</p>
            </div>
          </div>
          <button id="close-test-catalog-btn" className="icon-action-btn" onClick={onClose} aria-label="Close catalog">
            <X size={16} />
          </button>
        </header>

        {/* Top Domain Census Pills */}
        <div className="catalog-census-strip">
          <div className="census-badge">
            <span>Total Registered:</span> <b>{totalCount}</b>
          </div>
          <div className="census-badge">
            <span>Predictive ML:</span> <b>{predCount}</b>
          </div>
          <div className="census-badge">
            <span>Market Risk:</span> <b>{marketCount}</b>
          </div>
          <div className="census-badge">
            <span>Treasury:</span> <b>{treasuryCount}</b>
          </div>
        </div>

        {/* Search & Domain Filter Bar */}
        <div className="catalog-controls-box">
          <div className="catalog-search-input-wrap">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              id="catalog-search-input"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search tests by ID, name, or keyword…"
            />
            {searchQuery && (
              <button className="clear-search-btn" onClick={() => setSearchQuery('')}>
                <X size={12} />
              </button>
            )}
          </div>

          <div className="catalog-domain-tabs" role="tablist">
            <button
              className={`domain-tab ${domainFilter === 'all' ? 'active' : ''}`}
              onClick={() => { setDomainFilter('all'); setSelectedFamily(null) }}
            >
              All ({totalCount})
            </button>
            <button
              className={`domain-tab ${domainFilter === 'predictive_ml' ? 'active' : ''}`}
              onClick={() => { setDomainFilter('predictive_ml'); setSelectedFamily(null) }}
            >
              Predictive ML ({predCount})
            </button>
            <button
              className={`domain-tab ${domainFilter === 'quantitative_finance' ? 'active' : ''}`}
              onClick={() => { setDomainFilter('quantitative_finance'); setSelectedFamily(null) }}
            >
              Market Risk ({marketCount})
            </button>
            <button
              className={`domain-tab ${domainFilter === 'treasury' ? 'active' : ''}`}
              onClick={() => { setDomainFilter('treasury'); setSelectedFamily(null) }}
            >
              Treasury ({treasuryCount})
            </button>
          </div>

          {/* Family Filter Chips */}
          <div className="catalog-family-chips">
            <button
              className={`family-chip ${selectedFamily === null ? 'active' : ''}`}
              onClick={() => setSelectedFamily(null)}
            >
              All Families
            </button>
            {availableFamilies.map(fam => (
              <button
                key={fam}
                className={`family-chip ${selectedFamily === fam ? 'active' : ''}`}
                onClick={() => setSelectedFamily(fam === selectedFamily ? null : fam)}
              >
                {fam}
              </button>
            ))}
          </div>
        </div>

        {/* Tests List */}
        <div className="catalog-test-list">
          {filteredTests.length === 0 ? (
            <div className="catalog-empty-state">
              <CircleSlash2 size={24} className="text-muted" />
              <p>No tests match your search query.</p>
            </div>
          ) : (
            filteredTests.map(t => {
              const { status, reason } = getApplicability(t)
              const isApplicable = status === 'APPLICABLE'
              const isNotApplicable = status === 'NOT_APPLICABLE'

              return (
                <div key={t.testId} className={`test-entry-card ${isApplicable ? 'applicable' : ''}`}>
                  <div className="test-card-top-row">
                    <div className="test-id-and-family">
                      <code className="test-id-mono">{t.testId}</code>
                      <span className="test-family-badge">{t.family}</span>
                    </div>

                    <span
                      className={`applicability-pill ${
                        isApplicable ? 'pill-sage' : isNotApplicable ? 'pill-muted' : 'pill-default'
                      }`}
                    >
                      {isApplicable && <Check size={11} />}
                      {status}
                    </span>
                  </div>

                  <h4 className="test-name-title">{t.name}</h4>
                  <p className="test-desc-text">{t.description}</p>

                  <div className="test-applicability-reason">
                    <span className="reason-label">Evaluation:</span>
                    <span className="reason-text">{reason}</span>
                  </div>

                  {t.riskStripes && t.riskStripes.length > 0 && (
                    <div className="test-meta-strip">
                      <span className="meta-label">Risk Stripes:</span>
                      {t.riskStripes.map(s => (
                        <span key={s} className="meta-tag">{s}</span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>
      </aside>
    </div>
  )
}
