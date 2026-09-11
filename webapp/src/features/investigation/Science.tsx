import { useState, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'
import { display, label, object, type ScientificRecord } from './presentation'

export function CopyValue({ value, truncate = false }: { value: string; truncate?: boolean }) {
  const [copied, setCopied] = useState(false)
  return <span className="copy-value"><code title={value}>{truncate && value.length > 30 ? `${value.slice(0, 14)}…${value.slice(-8)}` : value}</code><button className="icon-button" aria-label="Copy value" title={copied ? 'Copied' : 'Copy value'} onClick={async () => { try { await navigator.clipboard.writeText(value); setCopied(true) } catch { setCopied(false) } }}>{copied ? <Check size={14}/> : <Copy size={14}/>}</button></span>
}
export function Empty({ title = 'Not supplied', children }: { title?: string; children?: ReactNode }) {
  return <div className="scientific-empty"><strong>{title}</strong><p>{children || 'This run does not supply this analytical output.'}</p></div>
}
export function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return <section className="science-section"><header><h2>{title}</h2>{note && <p>{note}</p>}</header>{children}</section>
}
export function ScoreStrip({ data }: { data: ScientificRecord }) {
  const entries = Object.entries(data).filter(([,v]) => ['number', 'string', 'boolean'].includes(typeof v))
  if (!entries.length) return <Empty title="Metrics not supplied"/>
  return <dl className="score-strip">{entries.map(([k,v]) => <div key={k}><dt>{label(k)}</dt><dd title={String(v)}>{typeof v === 'number' && !Number.isInteger(v) ? (Math.abs(v) < .0001 && v !== 0 ? v.toExponential(3) : v.toLocaleString('en-US', { maximumFractionDigits: 4 })) : display(v)}</dd></div>)}</dl>
}
export function ScientificTable({ rows }: { rows: ScientificRecord[] }) {
  if (!rows.length) return <Empty title="No records supplied"/>
  const keys = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  return <div className="science-table-scroll" tabIndex={0} role="region" aria-label="Scientific table"><table className="science-table"><thead><tr>{keys.map(k => <th key={k}>{label(k)}</th>)}</tr></thead><tbody>{rows.map((r,i) => <tr key={i}>{keys.map(k => <td key={k}>{r[k] && typeof r[k] === 'object' ? <details><summary>Inspect values</summary><DataValue value={r[k]}/></details> : <DataValue value={r[k]}/>}</td>)}</tr>)}</tbody></table></div>
}
export function DataValue({ value }: { value: unknown }) {
  if (value == null) return <span className="quiet">Not supplied</span>
  if (Array.isArray(value)) {
    if (!value.length) return <span className="quiet">None recorded</span>
    if (value.every(v => v && typeof v === 'object' && !Array.isArray(v))) return <ScientificTable rows={value}/>
    return <ol className="value-list">{value.map((v,i) => <li key={i}><DataValue value={v}/></li>)}</ol>
  }
  if (typeof value === 'object') return <dl className="data-definition">{Object.entries(value).map(([k,v]) => <div key={k}><dt>{label(k)}</dt><dd><DataValue value={v}/></dd></div>)}</dl>
  if (typeof value === 'string' && /^[a-f0-9]{40,}$/i.test(value)) return <CopyValue value={value} truncate/>
  return <span>{display(value)}</span>
}
export function DataBlock({ title, data, note }: { title: string; data: unknown; note?: string }) {
  const present = data != null && (typeof data !== 'object' || Object.keys(data).length > 0)
  return <Section title={title} note={note}>{present ? <DataValue value={data}/> : <Empty/>}</Section>
}
// Scaling below is solely coordinate layout of supplied observations, never scientific computation.
export function LineFigure({ series, xLabel, yLabel, unitSquare = false, reference = false }: { series: { name: string; x: number[]; y: number[] }[]; xLabel: string; yLabel: string; unitSquare?: boolean; reference?: boolean }) {
  const valid = series.filter(s => s.x.length > 0 && s.x.length === s.y.length && [...s.x,...s.y].every(Number.isFinite))
  if (!valid.length) return <Empty title="Curve not supplied"/>
  const xs = valid.flatMap(s => s.x), ys = valid.flatMap(s => s.y)
  const xmin = unitSquare ? 0 : Math.min(...xs), xmax = unitSquare ? 1 : Math.max(...xs)
  const ymin = unitSquare ? 0 : Math.min(...ys), ymax = unitSquare ? 1 : Math.max(...ys)
  const x = (v: number) => 60 + (v - xmin) / (xmax - xmin || 1) * 480
  const y = (v: number) => 260 - (v - ymin) / (ymax - ymin || 1) * 220
  return <figure className="line-figure"><svg viewBox="0 0 580 320" role="img" aria-label={`${yLabel} against ${xLabel}`}>
    {[0,.25,.5,.75,1].map(t => <g key={t}><line x1="60" x2="540" y1={40+t*220} y2={40+t*220} stroke="var(--line)"/><text x="50" y={44+t*220} textAnchor="end">{(ymax - t*(ymax-ymin)).toFixed(2)}</text><text x={60+t*480} y="281" textAnchor="middle">{(xmin+t*(xmax-xmin)).toFixed(unitSquare || xmax-xmin < 1 ? 2 : 0)}</text></g>)}
    {reference && <line x1="60" y1="260" x2="540" y2="40" stroke="var(--faint)" strokeDasharray="5 5"/>}
    {valid.map((s,i) => <g key={s.name} style={{color: i === 0 ? 'var(--accent)' : 'var(--sage)'}}><polyline points={s.x.map((v,j) => `${x(v)},${y(s.y[j])}`).join(' ')} fill="none" stroke="currentColor" strokeWidth="2.5"/>{s.x.map((v,j) => <circle key={j} cx={x(v)} cy={y(s.y[j])} r={s.x.length > 20 ? 1.5 : 3} fill="currentColor"><title>{`${s.name}: ${v}, ${s.y[j]}`}</title></circle>)}</g>)}
    <text x="300" y="310" textAnchor="middle">{xLabel}</text><text transform="translate(14 155) rotate(-90)" textAnchor="middle">{yLabel}</text>
  </svg><figcaption>{valid.map((s,i) => <span key={s.name}><i style={{background: i === 0 ? 'var(--accent)' : 'var(--sage)'}}/>{s.name}</span>)}{reference && <span>Dashed: identity reference</span>}</figcaption><details><summary>Inspect exact observations</summary><ScientificTable rows={valid.flatMap(s => s.x.map((v,i) => ({ series: s.name, [xLabel]: v, [yLabel]: s.y[i] })))}/></details></figure>
}
export function Bars({ data, name = 'Value' }: { data: ScientificRecord; name?: string }) {
  const entries = Object.entries(data).filter((e): e is [string, number] => typeof e[1] === 'number' && Number.isFinite(e[1])).sort((a,b) => b[1]-a[1])
  if (!entries.length) return <Empty/>
  const max = Math.max(...entries.map(([,v]) => Math.abs(v))) || 1
  return <div className="scientific-bars" aria-label={name}>{entries.map(([k,v]) => <div className="bar-row" key={k}><span>{k}</span><div className="bar-track"><i className={v < 0 ? 'negative' : ''} style={{width:`${Math.abs(v)/max*100}%`}}/></div><code title={String(v)}>{String(v)}</code></div>)}</div>
}
export function Matrix({ data, columns, title = 'Matrix' }: { data: unknown; columns?: string[]; title?: string }) {
  const o = object(data), names = columns ?? Object.keys(o)
  const matrix: unknown[][] = Array.isArray(data) ? data : names.map(r => names.map(c => o[r]?.[c]))
  if (!matrix.length) return <Empty/>
  return <div className="science-table-scroll"><table className="science-table heatmap" aria-label={title}><thead><tr><th></th>{names.map(n => <th key={n}>{n}</th>)}</tr></thead><tbody>{matrix.map((row,i) => <tr key={i}><th>{names[i] ?? i}</th>{row.map((v,j) => <td key={j} style={{background: typeof v === 'number' ? `rgba(83, 83, 145, ${.04+Math.min(1,Math.abs(v))*.24})` : undefined}} title={display(v)}>{typeof v === 'number' ? v.toFixed(3) : display(v)}</td>)}</tr>)}</tbody></table></div>
}
export function ConfusionMatrix({ data }: { data: ScientificRecord }) {
  const cm = object(data.confusion_matrix ?? data)
  if (!['tn','fp','fn','tp'].every(k => typeof cm[k] === 'number')) return <Empty title="Confusion matrix not supplied"/>
  return <div className="science-table-scroll"><table className="confusion-matrix" aria-label="Confusion matrix counts"><thead><tr><th>Actual / predicted</th><th>Negative</th><th>Positive</th></tr></thead><tbody>{[['Negative','tn','fp'],['Positive','fn','tp']].map(([row,...keys]) => <tr key={row}><th>{row}</th>{keys.map(k => <td key={k} className={k === 'tn' || k === 'tp' ? 'correct' : 'incorrect'}><strong>{cm[k]}</strong><span>{label(k)}</span></td>)}</tr>)}</tbody></table></div>
}
