---
name: emil-design-eng
description: A design engineering skill based on Emil Kowalski's articles and philosophy. Designed for designers and engineers to help them build better user interfaces with polished animations, interactions, and invisible details. Use this skill when reviewing UI polish, animations, transitions, or component interactions.
license: Based on emilkowalski/skill (github.com/emilkowalski/skill)
---

# Design Engineering Skill — by Emil Kowalski

This skill encodes a philosophy of trained taste: the belief that **unseen details compound to create interfaces that feel right**. Study why successful interfaces feel the way they do — don't just make things "work."

Use this skill on a **case-by-case basis** — not as an always-on rule set. It's most effective when reviewing animations, UI feedback, or component interactions.

When giving feedback or reviewing code, present changes as a **Markdown table** (Before / After) rather than bulleted lists, so the impact of each change is immediately visible.

---

## Core Philosophy

> "The best animations are the ones you don't notice."

Motion should feel **natural, purposeful, and invisible**. It should enhance the user experience without becoming decorative noise. Ask yourself: if a user explicitly notices this animation, is it overdone?

- **Perceived performance**: Animations should make the interface feel faster and more responsive.
- **Purposeful motion**: Every movement must have a reason — providing feedback, maintaining context, or guiding focus.
- **Invisible details**: The polish that separates great interfaces from good ones is often imperceptible consciously, yet deeply felt.

---

## Animation Decision Framework

### Frequency
- **High frequency** (repeated every few seconds or on every keystroke): No animation, or extremely subtle (opacity only).
- **Medium frequency** (on user action): Short, snappy animations (100–250ms).
- **Low frequency** (page load, modal open): Can afford more elaborate, longer transitions (300–600ms).

### Duration Guidelines
| Context | Duration |
|---|---|
| Micro-interactions (hovers, toggles, presses) | 150–250ms |
| Standard transitions (modals, panels, content) | 200–350ms |
| Complex orchestrations (page transitions, multi-step reveals) | 400–600ms total |
| Exit animations | Faster than entrance (e.g., enter 300ms → exit 200ms) |
| Stagger delays between items | 30–60ms tight |

**Rule**: Avoid animations longer than 1 second. If it takes longer, it should be a loading state, not a transition.

### Easing
- **`ease-out`** → Default for entrances. Natural deceleration (like a car arriving at a stop).
- **`ease-in`** → For exits. Natural acceleration as the element leaves.
- **`ease-in-out`** → Sparingly. For elements moving between two points while remaining on screen (e.g., dragging).
- **Spring physics** → Preferred for a more natural, fluid feel over duration-based easing. Tune stiffness and damping, not duration.
- **Never use CSS defaults** (`ease`, `linear`) for UI transitions. Always use custom `cubic-bezier()` curves.

### Property Selection
**Only animate `transform` and `opacity`.**
- These avoid layout and paint cycles, running off the main thread on the GPU.
- **Never animate**: `padding`, `margin`, `height`, `width`, `top`, `left` — these trigger expensive browser reflows ("jank").

---

## Component Building Principles

### Buttons & Interactive Elements
- Add **press/click feedback**: scale down slightly on `mousedown` (e.g., `scale(0.97)`), spring back on release.
- **Never scale from 0** — always start from a near-natural size (`scale(0.95)`, not `scale(0)`).
- Hover states should respond in **≤150ms**.
- Use `will-change: transform` only when you know an animation is imminent — don't apply globally.

### Popovers & Tooltips
- Apply a **tooltip delay** of 400–600ms before showing — avoids flashing on accidental hover.
- Popovers should animate from their **origin point** (where the trigger element is), not from the center of the screen.
- Use `@starting-style` for enter animations in CSS (native, no JS needed for simple cases).
- Exit faster than entry.

### Modals & Overlays
- Enter: fade in + slight scale up (`scale(0.95) → scale(1)`), 250–350ms ease-out.
- Exit: fade out, 150–200ms ease-in (faster than entry).
- Backdrop: fade separately, can be slightly slower than content.
- On mobile: slide up from bottom with spring physics.

### List & Content Reveals
- Stagger items with 30–50ms delays between each — tight enough to feel cohesive, not slow.
- One well-orchestrated page-load reveal creates more delight than scattered micro-animations everywhere.
- Use `animation-delay` with CSS `@keyframes` for simple staggered reveals — no JS needed.

### Scroll & Gesture Interactions
- Apply **momentum** after a flick gesture (velocity continues past the point of release).
- Apply **friction** to decelerate naturally.
- Apply **boundary damping**: elements should resist at boundaries and snap back with spring physics, not hard-stop.

### Blur & Masking Transitions
- Use blur as a transition property for content switches (blur out old content, blur in new).
- Use masking/clip-path animations for dramatic reveals.

---

## Performance Rules

| Rule | Detail |
|---|---|
| GPU-composited properties only | `transform`, `opacity` |
| Avoid layout triggers | No `height`, `width`, `margin`, `padding` animations |
| CSS for predetermined animations | Runs off main thread, survives heavy JS loads |
| JS for dynamic interactions | Use Framer Motion / Motion One for interruptible animations |
| `will-change` sparingly | Only apply just before animation, remove after |

---

## Accessibility

- **Always respect `prefers-reduced-motion`**: Disable or drastically reduce all animations when this media query is active.

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

- Never use motion as the **sole** way to convey information.
- Focus states must remain visible and not be obscured by animations.

---

## Code Review Format

When reviewing or suggesting UI improvements, always present changes as a table:

| Aspect | Before | After |
|---|---|---|
| Button press | No feedback | `scale(0.97)` on mousedown, spring back |
| Modal open | Instant appearance | `scale(0.95) → scale(1)`, 300ms ease-out |
| Tooltip | Appears immediately on hover | 500ms delay, then fades in |
| List items | All appear at once | Staggered 40ms fade-in per item |

This format makes the impact of each change immediately clear and reviewable.

---

## Quick Reference Checklist

Before shipping any UI with animation/interaction:

- [ ] Are all animations under 500ms (except orchestrated sequences)?
- [ ] Are only `transform` and `opacity` being animated?
- [ ] Are custom `cubic-bezier` curves used (no CSS defaults)?
- [ ] Do interactive elements have press feedback?
- [ ] Are tooltips/popovers delayed (400–600ms)?
- [ ] Do popovers animate from their trigger origin?
- [ ] Is `prefers-reduced-motion` respected?
- [ ] Do exit animations run faster than entrances?
- [ ] Are stagger delays tight (30–60ms)?
- [ ] Does motion feel purposeful, not decorative?
