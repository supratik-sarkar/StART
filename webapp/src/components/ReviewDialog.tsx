import { useEffect, useId, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'
export function ReviewDialog({ title, onClose, children, wide = false }: {title: string; onClose: () => void; children: ReactNode; wide?: boolean}) {
  const ref = useRef<HTMLDialogElement>(null), id = useId()
  const closeRef = useRef(onClose); closeRef.current = onClose
  useEffect(() => { const previous = document.activeElement as HTMLElement; const dialog = ref.current; dialog?.showModal(); return () => {dialog?.close();previous?.focus()} }, [])
  return <dialog ref={ref} className={`review-dialog ${wide ? 'wide' : ''}`} aria-labelledby={id} onCancel={e => {e.preventDefault();closeRef.current()}} onClick={e => {if(e.target === e.currentTarget) closeRef.current()}}><div className="dialog-surface"><header className="dialog-heading"><h2 id={id}>{title}</h2><button className="icon-button" aria-label="Close dialog" title="Close" onClick={onClose}><X size={18}/></button></header><div className="dialog-content">{children}</div></div></dialog>
}
