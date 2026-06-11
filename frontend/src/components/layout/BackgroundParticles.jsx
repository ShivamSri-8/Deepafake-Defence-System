import { useEffect, useRef } from 'react';

const BackgroundParticles = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // Aurora multi-hue particle configuration
        const particleCount = 50;
        const particles = [];
        const maxDistance = 140;

        // Palette for particles: cyan, blue, purple, green
        const palette = [
            { r: 6,   g: 200, b: 255 }, // cyan
            { r: 60,  g: 120, b: 255 }, // blue
            { r: 140, g: 60,  b: 255 }, // purple
            { r: 6,   g: 255, b: 160 }, // green
        ];

        for (let i = 0; i < particleCount; i++) {
            const col = palette[Math.floor(Math.random() * palette.length)];
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                radius: Math.random() * 1.5 + 0.5,
                r: col.r, g: col.g, b: col.b
            });
        }

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Subtle deep grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.012)';
            ctx.lineWidth = 1;
            const gridSize = 90;
            for (let x = 0; x < canvas.width; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }

            // Update and draw particles
            particles.forEach((p) => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, 0.5)`;
                ctx.fill();
            });

            // Draw connections with aurora color
            for (let i = 0; i < particleCount; i++) {
                for (let j = i + 1; j < particleCount; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < maxDistance) {
                        const alpha = (1 - dist / maxDistance) * 0.06;
                        const r = Math.round((particles[i].r + particles[j].r) / 2);
                        const g = Math.round((particles[i].g + particles[j].g) / 2);
                        const b = Math.round((particles[i].b + particles[j].b) / 2);
                        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                        ctx.lineWidth = 0.6;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: -2,
                pointerEvents: 'none',
                background: 'transparent'
            }}
        />
    );
};

export default BackgroundParticles;
