---
name: design-taste-frontend
description: Taste-Skill — gives your AI good taste. Stops the AI from generating boring, generic slop. The Anti-Slop Frontend Framework for AI Agents. Use for landing pages, portfolios, marketing sites, and UI redesigns. NOT for dashboards, data tables, or complex product flows.
author: Leonxlnx
source: https://github.com/Leonxlnx/taste-skill
version: v2 (experimental)
---

# Taste Skill — Anti-Slop Frontend Framework

> **Purpose**: Upgrade AI-built interfaces with stronger layout, typography, motion, and spacing — instead of boilerplate-looking UIs.

**Best for**: Landing pages, portfolios, marketing sites, creative projects, redesigns.
**Not for**: Dashboards, data tables, admin panels, complex multi-step product flows.

---

## Section 0: Brief Inference (Do This First)

Before writing any code, **read the room**. Analyze the following signals from the user's prompt or existing project:

| Signal | What to Look For |
|---|---|
| **Project type** | SaaS, portfolio, editorial, e-commerce, creative agency, startup |
| **Vibe words** | "minimalist," "brutalist," "glassy," "warm," "bold," "clean," "editorial" |
| **Audience** | Developers, creatives, enterprise buyers, consumers |
| **Reference signals** | Mentioned brands, existing color/font choices, UI screenshots |

Use this inference to set your design direction before touching any code. If vibe is unclear, default to **confident and editorial** rather than generic.

---

## Parametric Dials

Tune these per project (1–10 scale):

| Dial | Low (1–3) | Mid (4–6) | High (7–10) |
|---|---|---|---|
| **DESIGN_VARIANCE** | Centered, symmetric, safe | Grid-based, slight asymmetry | Bold asymmetric, diagonal, grid-breaking |
| **MOTION_INTENSITY** | Hover states only | Scroll reveals, transitions | Magnetic, parallax, orchestrated sequences |
| **VISUAL_DENSITY** | Spacious, generous whitespace | Balanced content/space | Information-dense, tighter grid |

Default starting values: `DESIGN_VARIANCE=6`, `MOTION_INTENSITY=5`, `VISUAL_DENSITY=4`

---

## Core Anti-Slop Directives

These are mandatory. No exceptions.

### Layout
- **Use CSS Grid, not repetitive flexbox math.** Named grid areas for complex layouts.
- Avoid centering everything — it's the lazy default. Use deliberate alignment.
- **Asymmetry is allowed and encouraged.** Perfect symmetry feels safe and forgettable.
- Use `min-h-[100dvh]` for full-viewport sections — not `100vh` (mobile browser chrome issue).
- Vary section heights — not every section should be the same viewport height.

### Typography
- **No Inter for creative/marketing projects.** Inter is a UI font, not a display face.
- Preferred alternatives for display: Geist, Outfit, Satoshi, Playfair Display, DM Serif Display, Syne, Epilogue (match to project vibe).
- Establish a strict **typographic scale** — hero, heading, subheading, body, caption each have distinct size AND weight AND tracking.
- Use `letter-spacing` intentionally — tight for large display text, slightly loose for all-caps labels.
- **No body text lighter than 0.65 opacity on dark backgrounds** — accessibility matters.

### Iconography
- Use **Phosphor Icons** (or Lucide) — not emojis in interfaces.
- Icons at consistent sizes: 16px body, 20px interactive, 24px feature, 32px+ decorative.
- Never use icons as primary content — they support text, not replace it.

### Color
- **Avoid purple-to-blue gradients.** They are the #1 signal of generic AI output.
- Use a dominant neutral + one sharp accent model.
- Background: Use `oklch()` or carefully crafted HSL — never pure `#000` or `#fff`.
- All color values should come from CSS custom properties: `--color-accent`, `--color-surface`, etc.

### Spacing
- **4px or 8px base unit.** No arbitrary values (13px, 22px, 37px).
- Padding between sections: min 80px on desktop, 48px on mobile.
- Card padding: min 24px. Never less than 16px.
- Gap between grid items: Use the base unit — 16px, 24px, 32px, 48px.

---

## Implementation Rules

### CSS Architecture
```css
/* Always use layers */
@layer reset, tokens, base, components, utilities;

/* Fluid typography */
:root {
  --text-hero: clamp(2.5rem, 6vw, 5rem);
  --text-heading: clamp(1.5rem, 3vw, 2.5rem);
  --text-body: clamp(0.9375rem, 1.5vw, 1.0625rem);
}

/* Color tokens — never hardcode */
:root {
  --color-bg: oklch(10% 0.01 240);
  --color-surface: oklch(14% 0.015 240);
  --color-accent: oklch(65% 0.2 280);
  --color-text: oklch(92% 0.01 240);
  --color-text-muted: oklch(60% 0.01 240);
}
```

### Animation Rules
- **Stagger entrance animations** for list items or card grids (40ms per item).
- Scroll-triggered reveals: `opacity: 0 → 1` + `translateY(20px → 0)`, 400ms ease-out.
- Hover on cards: `translateY(-2px)` + subtle shadow increase. No jarring scale.
- **Never animate layout properties** — only `transform` and `opacity`.

### Component Patterns
- **Hero**: Bold headline (5+ rem), supporting line, single CTA. Never center + stock photo.
- **Feature cards**: Vary sizing — don't make all cards identical heights.
- **CTA buttons**: Primary has background + strong contrast. Secondary is outlined or ghost. Tertiary is link-style.
- **Navigation**: On scroll, apply a subtle backdrop blur `backdrop-filter: blur(12px)` + reduced opacity background.

---

## What to Avoid (Anti-Slop Checklist)

Before shipping, confirm you have avoided:

- [ ] Purple-to-blue gradient hero backgrounds
- [ ] Three identical feature cards in a row
- [ ] Inter as the display font on creative projects  
- [ ] Emojis in the UI instead of proper icons
- [ ] `border-radius: 8px` applied universally to everything
- [ ] Box-shadow on every card (use spacing + background instead)
- [ ] Centered layout for everything on the page
- [ ] Stock photo hero background
- [ ] `100vh` (use `100dvh`)
- [ ] Hardcoded color values not in CSS variables
- [ ] Gray text at opacity < 0.5 on dark backgrounds
- [ ] Arbitrary spacing values not on the 4/8px grid

---

## Motion Guidance by Intensity Level

| Level | What to Implement |
|---|---|
| **Low (1–3)** | Hover color/opacity transitions (150ms), button press (scale 0.97) |
| **Mid (4–6)** | Scroll-triggered fade-ups, staggered list reveals, nav backdrop blur |
| **High (7–10)** | Magnetic hover on CTA buttons, parallax on hero elements, scroll-scrubbed animations |

Always respect `prefers-reduced-motion`:
```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Output Quality Standard

A passing output must:
1. Look unlike a template — a viewer should not be able to identify the "starter kit" it came from.
2. Have a clear typographic hierarchy visible within 2 seconds of viewing.
3. Use a coherent color system with no more than 3 primary hues.
4. Have at least one "moment of delight" — a micro-interaction, unusual layout choice, or typographic detail that surprises.
5. Work at 320px mobile width without horizontal scroll.
6. Pass WCAG AA contrast for all body text.
