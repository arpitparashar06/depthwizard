import React from 'react'
import { createRoot } from 'react-dom/client'

/* Fonts are SELF-HOSTED, not linked from a CDN, and that is deliberate: the
 * Docker image is built to run with no network at all, and a <link> to Google
 * Fonts would silently fall back to system faces in exactly the offline field
 * deployment this project is pitched at. @fontsource ships the woff2 files
 * through npm, so Vite bundles them and they are served from our own origin.
 *
 * Only the weights the stylesheet actually asks for are imported - every one
 * of these is a file in the build, so an unused weight is dead weight.
 *
 *   Inter             400 body · 500/600 labels · 700/800 headings
 *   Playfair Display  italic 400 only, and only twice (the hero flourish and
 *                     the empty-state line)
 *   JetBrains Mono    400 numbers · 600 small caps labels · 700 big readouts
 */
import '@fontsource/inter/latin-400.css'
import '@fontsource/inter/latin-500.css'
import '@fontsource/inter/latin-600.css'
import '@fontsource/inter/latin-700.css'
import '@fontsource/inter/latin-800.css'
import '@fontsource/playfair-display/latin-400-italic.css'
import '@fontsource/jetbrains-mono/latin-400.css'
import '@fontsource/jetbrains-mono/latin-600.css'
import '@fontsource/jetbrains-mono/latin-700.css'

import App from './App.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
