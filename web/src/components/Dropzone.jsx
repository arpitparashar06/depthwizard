import { useEffect, useRef, useState } from 'react'

const ACCEPT = '.png,.jpg,.jpeg,.tif,.tiff'
const KB = 1024

function human (bytes) {
  if (bytes < KB) return `${bytes} B`
  if (bytes < KB * KB) return `${(bytes / KB).toFixed(0)} KB`
  return `${(bytes / KB / KB).toFixed(1)} MB`
}

/* A .tif may or may not carry a CRS, and only the backend can say for sure -
 * load_image checks the coordinate system, the transform and the pixel width
 * before it commits. So the chip promises nothing it cannot know yet. */
function formatOf (name) {
  const ext = (name.split('.').pop() || '').toLowerCase()
  if (ext === 'tif' || ext === 'tiff') return { label: 'GeoTIFF', hint: 'metres if it carries a CRS' }
  return { label: ext.toUpperCase(), hint: 'relative surface, no metric scale' }
}

export default function Dropzone ({ file, onFile, disabled }) {
  const [over, setOver] = useState(false)
  const [preview, setPreview] = useState(null)
  const inputRef = useRef(null)

  // Browsers cannot decode a GeoTIFF, so those get the placeholder tile rather
  // than a broken image icon.
  useEffect(() => {
    if (!file || !/\.(png|jpe?g)$/i.test(file.name)) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const take = (f) => { if (f && !disabled) onFile(f) }

  if (file) {
    const fmt = formatOf(file.name)
    return (
      <div className="picked">
        {preview
          ? <img className="thumb" src={preview} alt="" />
          : <div className="thumb" style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--f-mono)', fontSize: 10, color: 'var(--faint)'
            }}>TIF</div>}
        <div className="meta">
          <span className="fname" title={file.name}>{file.name}</span>
          <span className="fsub">{fmt.label} · {human(file.size)} · {fmt.hint}</span>
        </div>
        <button className="clear" onClick={() => onFile(null)}
                disabled={disabled} title="Remove" aria-label="Remove file">×</button>
      </div>
    )
  }

  return (
    <div className={`drop${over ? ' over' : ''}`}
         onDragOver={(e) => { e.preventDefault(); setOver(true) }}
         onDragLeave={() => setOver(false)}
         onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0]) }}
         onClick={() => inputRef.current?.click()}>
      <input ref={inputRef} type="file" accept={ACCEPT} disabled={disabled}
             onChange={(e) => take(e.target.files?.[0])} />
      <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="1.5" aria-hidden="true">
        <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M3 15v3a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
      </svg>
      <p className="lead"><b>Drop an image</b> or click to browse</p>
      <p className="formats">PNG · JPG · GeoTIFF</p>
    </div>
  )
}
