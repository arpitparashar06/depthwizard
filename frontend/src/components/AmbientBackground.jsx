import { useEffect, useRef } from 'react'

export default function AmbientBackground () {
  const canvasRef = useRef(null)
  const spotRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const mouse = { x: null, y: null, radius: 140 }
    let width, height, particles = [], raf = 0

    // Ink, indigo, mint - the same three the rest of the page uses
    const COLORS = ['rgba(17, 17, 23, 0.20)', 'rgba(91, 79, 255, 0.24)', 'rgba(138, 237, 189, 0.30)']

    function resize () {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      const count = Math.min(Math.floor((width * height) / 14000), 110)
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 2 + 1,
        vx: (Math.random() - 0.5) * 0.55,
        vy: (Math.random() - 0.5) * 0.55,
        color: COLORS[Math.floor(Math.random() * COLORS.length)]
      }))
    }

    function onMove (e) {
      mouse.x = e.clientX
      mouse.y = e.clientY
      const s = spotRef.current
      if (s) {
        s.style.left = `${e.clientX}px`
        s.style.top = `${e.clientY}px`
        s.style.transform = 'translate(-50%, -50%) scale(1)'
      }
    }
    // Cursor left the window: forget it, or the dots keep fleeing a ghost
    function onLeave () {
      mouse.x = null
      mouse.y = null
      const s = spotRef.current
      if (s) s.style.transform = 'translate(-50%, -50%) scale(0)'
    }

    function frame () {
      ctx.clearRect(0, 0, width, height)
      // Pass 1: the links.
      const len = particles.length
      for (let a = 0; a < len; a++) {
        for (let b = a + 1; b < len; b++) {
          const dx = particles[a].x - particles[b].x
          const dy = particles[a].y - particles[b].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 115) {
            ctx.strokeStyle = `rgba(17, 17, 23, ${(1 - dist / 115) * 0.12})`
            ctx.lineWidth = 0.75
            ctx.beginPath()
            ctx.moveTo(particles[a].x, particles[a].y)
            ctx.lineTo(particles[b].x, particles[b].y)
            ctx.stroke()
          }
        }
      }
      // Pass 2: move (unless motion is reduced) and draw the dots
      for (let i = 0; i < len; i++) {
        const p = particles[i]
        if (!reduced) {
          p.x += p.vx; p.y += p.vy
          if (p.x < 0 || p.x > width) p.vx *= -1   // bounce off the edges
          if (p.y < 0 || p.y > height) p.vy *= -1
          if (mouse.x !== null && mouse.y !== null) {
            // Push out along the cursor->dot vector, strongest at the centre
            const dx = mouse.x - p.x, dy = mouse.y - p.y
            const dist = Math.sqrt(dx * dx + dy * dy)
            if (dist < mouse.radius && dist > 0) {
              const f = (mouse.radius - dist) / mouse.radius
              p.x -= (dx / dist) * f * 3.2
              p.y -= (dy / dist) * f * 3.2
            }
          }
        }
        ctx.fillStyle = p.color
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      }
      if (!reduced) raf = requestAnimationFrame(frame)
    }

    resize()
    frame()
    window.addEventListener('resize', resize)
    if (!reduced) {
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseleave', onLeave)
    }

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseleave', onLeave)
    }
  }, [])

  return (
    <>
      <canvas id="interactive-canvas" ref={canvasRef} />
      <div className="spotlight-circle" ref={spotRef} />
    </>
  )
}
