import React, { lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import App from './app/App'
import './design-system/tokens.css'
import './design-system/workstation.css'
import './design-system/science.css'
const Preview = import.meta.env.DEV ? lazy(() => import('./dev/UxPreview')) : null
const previewRequested = location.pathname === '/dev/ux-preview'
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode>{previewRequested ? Preview ? <Suspense fallback={<p>Loading preview gallery…</p>}><Preview/></Suspense> : <p>This development route is unavailable.</p> : <App/>}</React.StrictMode>)
