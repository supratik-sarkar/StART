import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './app/App'
import './design-system/tokens.css'
import './design-system/workstation.css'
import './design-system/science.css'
const unavailablePreviewPath = !import.meta.env.DEV && location.pathname === '/dev/ux-preview'
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode>{unavailablePreviewPath ? <p>This development route is unavailable.</p> : <App/>}</React.StrictMode>)
