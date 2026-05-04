import React, { useState, useEffect, useRef } from 'react';

const EPOCHS = [
  {
    id: 0,
    title: '1. The Informational Substrate',
    subtitle: 'Pre-Spacetime / Planck Scale',
    description:
      'The universe is not a spatial container, but a fluid mathematical medium. Physical locality is informational commutativity.',
    metrics: { phase: '0.00 rad', stiffness: 'Maximum', hRatio: 'N/A', activeProtocol: 'State Functor F' },
    color: '#06b6d4'
  },
  {
    id: 1,
    title: '2. The Shredding Phase',
    subtitle: 'Quark Degradation / High-Energy',
    description:
      'Under extreme density, adiabatic evolution breaks down and matter transitions toward an informational code-state.',
    metrics: { phase: '1.57 rad (π/2)', stiffness: 'Fracturing', hRatio: "0.92 (Smith's Limit)", activeProtocol: 'Grey Hole Kernel' },
    color: '#ef4444'
  },
  {
    id: 2,
    title: '3. Spiral Galaxy Formation',
    subtitle: 'Thermal Time & The Mark 1 Attractor',
    description:
      'Large-scale structures self-organize through informational gradients with regulated thermal dissipation.',
    metrics: { phase: '0.00 rad', stiffness: 'Localized', hRatio: '0.35 (Mark 1 Attractor)', activeProtocol: "Samson's Law V2" },
    color: '#a855f7'
  },
  {
    id: 3,
    title: '4. Black Hole Mergers',
    subtitle: 'Topological Tunneling & RCOD Flux',
    description:
      'Merger dynamics are represented as strong informational flux and metric reconciliation events.',
    metrics: { phase: 'Tunneling', stiffness: 'Infinite', hRatio: 'Fluctuating', activeProtocol: 'Entanglement Router' },
    color: '#f59e0b'
  },
  {
    id: 4,
    title: '5. The Harmonic Horizon',
    subtitle: 'Beyond the Current Trajectory',
    description:
      'The cosmos is modeled as a recursive harmonic architecture with continuous algorithmic retuning.',
    metrics: { phase: 'Locked', stiffness: 'Harmonic', hRatio: '0.349 (Perfected)', activeProtocol: 'Zero-Point Harmonic Collapse' },
    color: '#10b981'
  }
];

const OmegaUniverseApp = () => {
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const particles = useRef([]);
  const time = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = canvas.clientWidth;
    let height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;

    const numParticles = 800;
    particles.current = Array.from({ length: numParticles }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: 0,
      vy: 0,
      phase: Math.random() * Math.PI * 2
    }));

    const handleResize = () => {
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width;
      canvas.height = height;
    };

    window.addEventListener('resize', handleResize);

    const render = () => {
      time.current += 0.01;
      ctx.fillStyle = 'rgba(9, 9, 11, 0.2)';
      ctx.fillRect(0, 0, width, height);

      const epoch = EPOCHS[currentEpoch];
      const cx = width / 2;
      const cy = height / 2;

      particles.current.forEach((p) => {
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.hypot(dx, dy) + 1;

        p.vx += -dx * 0.0005 + Math.sin(time.current + p.phase) * 0.01;
        p.vy += -dy * 0.0005 + Math.cos(time.current + p.phase) * 0.01;

        if (currentEpoch === 1) {
          p.vx += (Math.random() - 0.5) * 0.6;
          p.vy += (Math.random() - 0.5) * 0.6;
        }

        if (currentEpoch === 2) {
          const tx = -dy / dist;
          const ty = dx / dist;
          p.vx += tx * 0.03;
          p.vy += ty * 0.03;
        }

        p.vx *= 0.97;
        p.vy *= 0.97;
        p.x = (p.x + p.vx + width) % width;
        p.y = (p.y + p.vy + height) % height;

        ctx.beginPath();
        ctx.fillStyle = `${epoch.color}${currentEpoch === 4 ? 'cc' : '99'}`;
        ctx.arc(p.x, p.y, currentEpoch === 4 ? 2.2 : 1.5, 0, Math.PI * 2);
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationRef.current);
    };
  }, [currentEpoch]);

  const activeData = EPOCHS[currentEpoch];

  return (
    <div className="flex flex-col h-screen w-full bg-zinc-950 text-slate-200 font-sans overflow-hidden">
      <header className="absolute top-0 left-0 w-full p-6 z-10 bg-gradient-to-b from-zinc-950 to-transparent pointer-events-none">
        <h1 className="text-3xl font-light tracking-widest text-white uppercase drop-shadow-md">
          Omega Protocol <span className="font-bold text-zinc-400 text-sm align-top ml-1">v3.1</span>
        </h1>
      </header>

      <div className="relative flex-grow w-full h-full">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full block" />

        <div className="absolute inset-0 flex flex-col md:flex-row justify-between items-end p-6 md:p-10 z-20 pointer-events-none">
          <div className="w-full md:w-1/3 bg-zinc-950/80 backdrop-blur-md border border-zinc-800 rounded-xl p-6 pointer-events-auto mb-6 md:mb-0">
            <h2 className="text-2xl font-bold mb-1" style={{ color: activeData.color }}>{activeData.title}</h2>
            <h3 className="text-sm font-mono text-zinc-400 mb-4 tracking-wider uppercase">{activeData.subtitle}</h3>
            <p className="text-sm text-zinc-300 leading-relaxed mb-6 font-light">{activeData.description}</p>
          </div>

          <div className="w-full md:w-1/2 bg-zinc-950/80 backdrop-blur-md border border-zinc-800 rounded-xl p-6 pointer-events-auto flex flex-col justify-center">
            <input
              type="range"
              min="0"
              max="4"
              step="1"
              value={currentEpoch}
              onChange={(e) => setCurrentEpoch(parseInt(e.target.value, 10))}
              className="w-full"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default OmegaUniverseApp;
