import { useState } from 'react'

export interface BarDatum {
  label: string
  value: number
  percentage?: number
  color?: string
}

/**
 * Interactive Target Distribution Bar Chart with tooltips.
 */
export function TargetDistributionBar({
  data,
  total,
  height = 140,
}: {
  data: BarDatum[]
  total: number
  height?: number
}) {
  const [hovered, setHovered] = useState<BarDatum | null>(null)
  const maxVal = Math.max(...data.map(d => d.value), 1)

  return (
    <div className="chart-container" style={{ position: 'relative', width: '100%' }}>
      {hovered && (
        <div
          className="chart-tooltip"
          style={{
            position: 'absolute',
            top: 4,
            right: 8,
            background: 'var(--ink)',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '6px',
            fontSize: '11px',
            pointerEvents: 'none',
            zIndex: 10,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <strong>{hovered.label}</strong>: {hovered.value.toLocaleString()} ({hovered.percentage ?? ((hovered.value / total) * 100).toFixed(1)}%)
        </div>
      )}
      <svg width="100%" height={height} viewBox="0 0 360 140" preserveAspectRatio="none" style={{ overflow: 'visible' }}>
        <line x1="40" y1="110" x2="350" y2="110" stroke="var(--line)" strokeWidth="1" />
        {data.map((d, i) => {
          const barWidth = 44
          const spacing = 110
          const x = 70 + i * spacing
          const barHeight = Math.max(8, (d.value / maxVal) * 85)
          const y = 110 - barHeight
          const isHovered = hovered?.label === d.label
          const fillColor = d.color || (i === 1 ? 'var(--accent)' : 'var(--line-strong)')

          return (
            <g
              key={d.label}
              onMouseEnter={() => setHovered(d)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                rx={5}
                fill={fillColor}
                opacity={isHovered ? 1 : 0.85}
                stroke={isHovered ? 'var(--ink)' : 'transparent'}
                strokeWidth={isHovered ? 1.5 : 0}
                style={{ transition: 'all 0.15s ease' }}
              />
              <text
                x={x + barWidth / 2}
                y={y - 6}
                textAnchor="middle"
                fontSize="11"
                fontWeight="700"
                fill="var(--ink)"
              >
                {d.value.toLocaleString()}
              </text>
              <text
                x={x + barWidth / 2}
                y="126"
                textAnchor="middle"
                fontSize="11"
                fill="var(--muted)"
              >
                {d.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/**
 * Interactive Correlation Heatmap Grid.
 */
export function CorrelationHeatmap({
  columns,
  matrix,
}: {
  columns: string[]
  matrix: number[][]
}) {
  const [hoveredCell, setHoveredCell] = useState<{ row: string; col: string; val: number } | null>(null)
  const n = columns.length
  if (n === 0 || matrix.length === 0) return <div className="text-muted text-center py-4">No correlation matrix available.</div>

  const cellSize = Math.min(36, Math.floor(320 / n))

  return (
    <div className="correlation-heatmap-wrapper" style={{ position: 'relative' }}>
      {hoveredCell && (
        <div
          className="chart-tooltip"
          style={{
            position: 'absolute',
            top: -24,
            right: 0,
            background: 'var(--ink)',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '6px',
            fontSize: '11px',
            zIndex: 10,
          }}
        >
          {hoveredCell.row} ↔ {hoveredCell.col}: <strong>r = {hoveredCell.val.toFixed(3)}</strong>
        </div>
      )}
      <div style={{ overflowX: 'auto', paddingBottom: '8px' }}>
        <table style={{ borderCollapse: 'collapse', margin: '0 auto', fontSize: '10px' }}>
          <thead>
            <tr>
              <th style={{ width: 60, padding: 4 }}></th>
              {columns.map(c => (
                <th
                  key={c}
                  style={{
                    width: cellSize,
                    padding: '2px 4px',
                    textAlign: 'center',
                    fontWeight: 600,
                    color: 'var(--muted)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: cellSize,
                  }}
                  title={c}
                >
                  {c.length > 7 ? c.slice(0, 6) + '…' : c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowCol, rIdx) => (
              <tr key={rowCol}>
                <td
                  style={{
                    padding: '2px 6px',
                    textAlign: 'right',
                    fontWeight: 600,
                    color: 'var(--muted)',
                    whiteSpace: 'nowrap',
                  }}
                  title={rowCol}
                >
                  {rowCol.length > 8 ? rowCol.slice(0, 7) + '…' : rowCol}
                </td>
                {columns.map((colCol, cIdx) => {
                  const val = matrix[rIdx]?.[cIdx] ?? 0
                  const absVal = Math.abs(val)
                  // Color interpolation based on absolute correlation
                  const bg = rIdx === cIdx
                    ? 'rgba(92, 91, 214, 0.25)'
                    : val > 0
                    ? `rgba(92, 91, 214, ${Math.max(0.04, absVal * 0.75)})`
                    : `rgba(181, 69, 69, ${Math.max(0.04, absVal * 0.75)})`

                  const isHovered = hoveredCell?.row === rowCol && hoveredCell?.col === colCol

                  return (
                    <td
                      key={colCol}
                      onMouseEnter={() => setHoveredCell({ row: rowCol, col: colCol, val })}
                      onMouseLeave={() => setHoveredCell(null)}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        textAlign: 'center',
                        backgroundColor: bg,
                        border: isHovered ? '1.5px solid var(--ink)' : '1px solid var(--surface)',
                        cursor: 'pointer',
                        fontWeight: absVal > 0.4 ? 700 : 400,
                        color: absVal > 0.6 ? '#fff' : 'var(--ink)',
                        transition: 'border 0.1s ease',
                      }}
                      title={`${rowCol} ↔ ${colCol}: ${val.toFixed(3)}`}
                    >
                      {val.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/**
 * Interactive Horizontal Bar Chart for weights, exposures, or correlations.
 */
export function HorizontalBarChart({
  items,
  valueKey = 'value',
  labelKey = 'label',
  maxItems = 10,
  unit = '',
}: {
  items: Array<Record<string, any>>
  valueKey?: string
  labelKey?: string
  maxItems?: number
  unit?: string
}) {
  const [hovered, setHovered] = useState<any | null>(null)
  const displayItems = items.slice(0, maxItems)
  const maxAbsVal = Math.max(...displayItems.map(d => Math.abs(Number(d[valueKey]) || 0)), 0.001)

  return (
    <div className="horizontal-bar-chart" style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11px' }}>
      {displayItems.map(item => {
        const val = Number(item[valueKey]) || 0
        const pct = Math.min(100, (Math.abs(val) / maxAbsVal) * 100)
        const label = String(item[labelKey] || '')
        const isHovered = hovered === item
        const isNegative = val < 0

        return (
          <div
            key={label}
            onMouseEnter={() => setHovered(item)}
            onMouseLeave={() => setHovered(null)}
            style={{
              display: 'grid',
              gridTemplateColumns: '90px 1fr 65px',
              alignItems: 'center',
              gap: '8px',
              padding: '2px 4px',
              borderRadius: '4px',
              backgroundColor: isHovered ? 'var(--canvas-2)' : 'transparent',
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                textAlign: 'right',
                color: 'var(--muted)',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              title={label}
            >
              {label}
            </span>
            <div style={{ height: '14px', background: 'var(--canvas)', borderRadius: '3px', overflow: 'hidden', position: 'relative' }}>
              <div
                style={{
                  height: '100%',
                  width: `${pct}%`,
                  backgroundColor: isNegative ? 'var(--red)' : 'var(--accent)',
                  borderRadius: '3px',
                  transition: 'width 0.2s ease',
                  opacity: isHovered ? 1 : 0.85,
                }}
              />
            </div>
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: '10px', textAlign: 'right', fontWeight: 600, color: 'var(--ink)' }}>
              {val.toFixed(4)}{unit}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/**
 * Interactive SVG Line Path for short rate or timeseries.
 */
export function RatePathChart({
  points,
  height = 120,
}: {
  points: Array<{ step: number; rate: number }>
  height?: number
}) {
  const [hovered, setHovered] = useState<{ step: number; rate: number } | null>(null)
  if (!points || points.length === 0) return null

  const rates = points.map(p => p.rate)
  const minRate = Math.min(...rates)
  const maxRate = Math.max(...rates)
  const range = maxRate - minRate || 0.01

  const width = 400
  const padLeft = 40
  const padRight = 20
  const padTop = 15
  const padBottom = 25
  const drawWidth = width - padLeft - padRight
  const drawHeight = height - padTop - padBottom

  const coords = points.map((p, idx) => {
    const x = padLeft + (idx / (points.length - 1)) * drawWidth
    const y = padTop + (1 - (p.rate - minRate) / range) * drawHeight
    return { x, y, p }
  })

  const pathD = coords.reduce((acc, pt, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`, '')

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {hovered && (
        <div
          className="chart-tooltip"
          style={{
            position: 'absolute',
            top: 0,
            right: 8,
            background: 'var(--ink)',
            color: '#fff',
            padding: '3px 8px',
            borderRadius: '5px',
            fontSize: '11px',
            fontFamily: 'ui-monospace, monospace',
            zIndex: 10,
          }}
        >
          t = {hovered.step}: <strong>r = {(hovered.rate * 100).toFixed(2)}%</strong>
        </div>
      )}
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ overflow: 'visible' }}>
        {/* Horizontal axis grid */}
        <line x1={padLeft} y1={padTop + drawHeight} x2={padLeft + drawWidth} y2={padTop + drawHeight} stroke="var(--line)" strokeWidth="1" />
        <line x1={padLeft} y1={padTop} x2={padLeft + drawWidth} y2={padTop} stroke="var(--line)" strokeWidth="0.5" strokeDasharray="3 3" />
        
        {/* Y Axis labels */}
        <text x={padLeft - 6} y={padTop + 4} textAnchor="end" fontSize="9" fill="var(--muted)">
          {(maxRate * 100).toFixed(1)}%
        </text>
        <text x={padLeft - 6} y={padTop + drawHeight} textAnchor="end" fontSize="9" fill="var(--muted)">
          {(minRate * 100).toFixed(1)}%
        </text>

        {/* Path line */}
        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

        {/* Interactive hover points */}
        {coords.map((c, idx) => (
          <circle
            key={idx}
            cx={c.x}
            cy={c.y}
            r={hovered?.step === c.p.step ? 4 : 2}
            fill={hovered?.step === c.p.step ? 'var(--ink)' : 'var(--accent)'}
            stroke="#fff"
            strokeWidth="1"
            onMouseEnter={() => setHovered(c.p)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer', transition: 'r 0.1s' }}
          />
        ))}
      </svg>
    </div>
  )
}
