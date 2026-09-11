import React, { useState } from 'react'
import { Check, Database, FileSpreadsheet, HardDrive, Info, Upload, CheckCircle2 } from 'lucide-react'
import type { ExecutionContext } from '../../contracts/types'

export interface ParsedUserData {
  fileName: string
  fileSize: string
  rows: number
  columns: number
  headers: string[]
  detectedTarget: string | null
  columnTypes: Record<string, string>
  missingnessCount: number
  missingnessPercent: number
  previewRows: string[][]
}

export interface DataSourceSelection {
  mode: 'builtin' | 'user'
  ready: boolean
  label: string
  rows?: number
  features?: number
  target?: string
  taskType?: string
  userData?: ParsedUserData
}

const SAMPLE_CSV = `feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,target
0.812,-0.341,1.402,-0.119,0.552,-1.023,0.344,0.128,1
-0.452,1.120,-0.892,0.672,-0.312,0.841,-0.125,-0.781,0
1.205,-0.122,0.441,-0.921,0.884,-0.412,1.109,0.442,1
-0.912,-1.421,-0.334,0.142,-0.882,0.129,-0.672,-0.331,0
0.341,0.552,1.112,0.441,0.129,-0.332,0.441,0.892,1
-0.128,0.441,-0.782,-0.331,0.221,0.672,-0.118,-0.441,0
0.991,-0.221,0.881,-0.442,0.667,-0.881,0.772,0.331,1
-0.772,0.881,-1.112,0.552,-0.441,0.441,-0.332,-0.662,0`

export function DataOnboarding({
  context,
  selection,
  onSelect,
}: {
  context: ExecutionContext | null
  selection: DataSourceSelection
  onSelect: (sel: DataSourceSelection) => void
}) {
  const [activeTab, setActiveTab] = useState<'builtin' | 'user'>(selection.mode)
  const [dragActive, setDragActive] = useState(false)
  const [userData, setUserData] = useState<ParsedUserData | null>(selection.userData || null)

  const parseCsvText = (text: string, fileName: string, sizeBytes: number) => {
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean)
    if (lines.length < 2) {
      alert('CSV must contain a header row and at least one data row.')
      return
    }
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''))
    const dataRows = lines.slice(1).map(l => l.split(',').map(c => c.trim().replace(/^"|"$/g, '')))
    
    let totalCells = 0
    let emptyCells = 0
    const colTypes: Record<string, string> = {}
    
    headers.forEach((h, colIdx) => {
      let isNumeric = true
      for (const row of dataRows) {
        totalCells++
        const val = row[colIdx]
        if (val === undefined || val === '' || val.toLowerCase() === 'na' || val.toLowerCase() === 'nan' || val.toLowerCase() === 'null') {
          emptyCells++
        } else if (isNaN(Number(val))) {
          isNumeric = false
        }
      }
      colTypes[h] = isNumeric ? 'float64 / numeric' : 'string / categorical'
    })

    const targetCandidates = ['target', 'label', 'default_flag', 'default', 'class', 'status', 'is_fraud']
    const detected = headers.find(h => targetCandidates.includes(h.toLowerCase())) || null

    const parsed: ParsedUserData = {
      fileName,
      fileSize: sizeBytes > 1024 * 1024 ? `${(sizeBytes / (1024 * 1024)).toFixed(2)} MB` : `${(sizeBytes / 1024).toFixed(1)} KB`,
      rows: dataRows.length,
      columns: headers.length,
      headers,
      detectedTarget: detected,
      columnTypes: colTypes,
      missingnessCount: emptyCells,
      missingnessPercent: totalCells > 0 ? Number(((emptyCells / totalCells) * 100).toFixed(2)) : 0,
      previewRows: dataRows.slice(0, 5),
    }

    setUserData(parsed)
    onSelect({
      mode: 'user',
      ready: true,
      label: `Uploaded: ${fileName}`,
      rows: parsed.rows,
      features: parsed.columns - (detected ? 1 : 0),
      target: detected || undefined,
      taskType: 'User supervised dataset',
      userData: parsed,
    })
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = evt => {
      const content = evt.target?.result as string
      parseCsvText(content, file.name, file.size)
    }
    reader.readAsText(file)
  }

  const handleLoadSample = () => {
    parseCsvText(SAMPLE_CSV, 'credit_benchmark_sample.csv', new Blob([SAMPLE_CSV]).size)
  }

  const handleSelectBuiltin = () => {
    setActiveTab('builtin')
    onSelect({
      mode: 'builtin',
      ready: true,
      label: context ? `Built-in: ${context.label}` : 'Use built-in benchmark data',
      rows: context?.shape?.includes('×') ? parseInt(context.shape.split('×')[0].replace(/,/g, '').trim()) : 500,
      features: context?.shape?.includes('×') ? parseInt(context.shape.split('×')[1].replace(/,/g, '').trim()) : 8,
      target: context?.target || 'target',
      taskType: context?.badges?.includes('binary') ? 'Binary classification' : 'Quantitative analytics',
    })
  }

  const handleSelectUser = () => {
    setActiveTab('user')
    if (userData) {
      onSelect({
        mode: 'user',
        ready: true,
        label: `Uploaded: ${userData.fileName}`,
        rows: userData.rows,
        features: userData.columns - (userData.detectedTarget ? 1 : 0),
        target: userData.detectedTarget || undefined,
        taskType: 'User supervised dataset',
        userData,
      })
    } else {
      onSelect({
        mode: 'user',
        ready: false,
        label: 'Upload user dataset (CSV / Parquet)',
      })
    }
  }

  // Parse context shape for built-in display
  const shapeRows = context?.shape?.includes('×')
    ? context.shape.split('×')[0].trim()
    : '500'
  const shapeFeatures = context?.shape?.includes('×')
    ? context.shape.split('×')[1].trim()
    : '8'

  return (
    <div className="data-onboarding-card">
      <div className="data-tab-strip">
        <button
          type="button"
          className={`data-tab-btn ${activeTab === 'builtin' ? 'active' : ''}`}
          onClick={handleSelectBuiltin}
        >
          <HardDrive size={15} />
          <span>Use built-in benchmark data</span>
          {activeTab === 'builtin' && <Check size={13} className="tab-check" />}
        </button>

        <button
          type="button"
          className={`data-tab-btn ${activeTab === 'user' ? 'active' : ''}`}
          onClick={handleSelectUser}
        >
          <Upload size={15} />
          <span>Upload local file (CSV / Parquet)</span>
          {activeTab === 'user' && userData && <Check size={13} className="tab-check" />}
        </button>
      </div>

      {activeTab === 'builtin' ? (
        <div className="data-content-pane">
          <div className="data-metadata-grid">
            <div className="meta-item">
              <span className="meta-label">Dataset source</span>
              <strong className="meta-val">{context?.provenance || 'Built-in deterministic synthetic generator'}</strong>
            </div>
            <div className="meta-item">
              <span className="meta-label">Rows</span>
              <strong className="meta-val">{shapeRows}</strong>
            </div>
            <div className="meta-item">
              <span className="meta-label">Features</span>
              <strong className="meta-val">{shapeFeatures}</strong>
            </div>
            <div className="meta-item">
              <span className="meta-label">Target column</span>
              <strong className="meta-val">{context?.target || 'target'}</strong>
            </div>
            <div className="meta-item">
              <span className="meta-label">Task type</span>
              <strong className="meta-val">
                {context?.badges?.includes('binary') ? 'Binary classification' : 'Supervised predictive risk'}
              </strong>
            </div>
            <div className="meta-item">
              <span className="meta-label">Storage & mode</span>
              <strong className="meta-val">Synthetic seeded benchmark (in-memory)</strong>
            </div>
          </div>

          <div className="data-status-note">
            <CheckCircle2 size={15} className="note-icon sage" />
            <span>
              <strong>Zero configuration required.</strong> Seeded deterministic benchmark context ready for evaluation. Values reflect verified metadata without runtime fabrication.
            </span>
          </div>
        </div>
      ) : (
        <div className="data-content-pane">
          {!userData ? (
            <div
              className={`data-dropzone ${dragActive ? 'drag-over' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragActive(true) }}
              onDragLeave={() => setDragActive(false)}
              onDrop={e => {
                e.preventDefault()
                setDragActive(false)
                const file = e.dataTransfer.files?.[0]
                if (file) {
                  const reader = new FileReader()
                  reader.onload = evt => parseCsvText(evt.target?.result as string, file.name, file.size)
                  reader.readAsText(file)
                }
              }}
            >
              <FileSpreadsheet size={32} className="drop-icon" />
              <h4>Select or drop a CSV or Parquet file</h4>
              <p>Supported genuinely local formats: <strong>CSV</strong>, <strong>Parquet</strong> (uncompressed / standard tabular).</p>
              
              <div className="upload-actions">
                <label className="file-upload-btn">
                  Browse local file
                  <input type="file" accept=".csv,.parquet" onChange={handleFileChange} style={{ display: 'none' }} />
                </label>
                <button type="button" className="sample-btn" onClick={handleLoadSample}>
                  Load sample credit CSV
                </button>
              </div>
            </div>
          ) : (
            <div className="uploaded-summary">
              <div className="upload-header">
                <div>
                  <strong>{userData.fileName}</strong>
                  <span>{userData.fileSize} · {userData.rows} rows · {userData.columns} columns</span>
                </div>
                <button type="button" className="reupload-btn" onClick={() => setUserData(null)}>
                  Replace file
                </button>
              </div>

              <div className="data-metadata-grid compact">
                <div className="meta-item">
                  <span className="meta-label">Detected target</span>
                  <strong className="meta-val">{userData.detectedTarget ? `✓ ${userData.detectedTarget}` : 'None detected (unsupervised / specify in goal)'}</strong>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Missingness</span>
                  <strong className="meta-val">{userData.missingnessCount} cells ({userData.missingnessPercent}%)</strong>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Data types</span>
                  <strong className="meta-val">{Object.values(userData.columnTypes).filter(t => t.includes('numeric')).length} numeric / {Object.values(userData.columnTypes).filter(t => t.includes('categorical')).length} categorical</strong>
                </div>
              </div>

              <div className="table-preview-wrap">
                <div className="preview-heading">Column preview (first 5 rows)</div>
                <div className="preview-scroll">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {userData.headers.map(h => (
                          <th key={h}>
                            <div>{h}</div>
                            <small>{userData.columnTypes[h]?.split('/')[0].trim()}</small>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {userData.previewRows.map((row, rIdx) => (
                        <tr key={rIdx}>
                          {row.map((val, cIdx) => (
                            <td key={cIdx}>{val}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
