---
name: impeccable
description: The design language that makes your AI harness better at design. Overrides generic AI aesthetics with professional design principles — OKLCH colors, vertical typographic rhythm, intentional spatial design, and an anti-pattern library. Use this when building any UI and want to avoid "AI slop" aesthetics.
author: pbakaus (Paul Bakaus)
source: https://github.com/pbakaus/impeccable
---

# Impeccable — AI Design Language

Impeccable is a design language and set of constraints that force professional design principles onto AI-generated UI. Most AI models default to predictable patterns: Inter font, purple-to-blue gradients, nested cards, gray text on colored backgrounds. This skill overrides those defaults.

## Core Philosophy

Design is not decoration — it is function made visible. Every visual decision should be intentional, purposeful, and rooted in hierarchy and clarity. Ask before you code: *What is the user trying to accomplish? What should they see first? What is the emotional register of this interface?*

---

## Initialization

When starting a new project, run through this mental checklist before writing any code:

1. **Brand Voice**: What is the product's personality? (e.g., clinical/precise, playful/warm, editorial/authoritative)
2. **Audience**: Who uses this? What do they expect vs. what will surprise them positively?
3. **Primary Action**: What is the single most important thing a user should do on this screen?
4. **Constraint**: What must you NOT do? (Identify 2–3 anti-goals for this specific project)

Document these in a `DESIGN.md` or `PRODUCT.md` in the project root for persistent context.

---

## Design Principles

### Typography
- **Never use Inter, Roboto, or Arial as a display face.** These are system defaults, not design choices.
- Establish a clear **typographic scale**: hero → heading → subheading → body → caption. Each step should have a clear visual difference in size, weight, and/or color.
- **Vertical rhythm**: Line heights and spacing should create a consistent grid. Body text at 1.5–1.7 line-height. Headings at 1.1–1.2.
- Pair a **distinctive display font** with a functional body font. Don't match two decorative fonts.
- Use `font-feature-settings` to enable ligatures, oldstyle numerals, and contextual alternates where available.

### Color
- **Use OKLCH color space** for consistent perceptual lightness across hues. Avoid HSL for anything beyond prototyping.
- **Dominant + accent model**: One dominant neutral palette, one sharp accent. Don't distribute 4+ colors equally.
- Dark mode is not just inverting colors — it requires a separate color token system with lower contrast ratios for large surfaces and higher contrast for text.
- **No purple-to-blue gradients.** No cyan glow effects. These are overused to the point of meaninglessness.
- Background colors should have subtle warmth or coolness — pure `#000000` and `#ffffff` are almost never right.

### Layout & Spacing
- **Grid-first**: Design on a grid. 12 columns for desktop, 4 for mobile. Sidebar layouts should use named grid areas.
- Spacing should follow a **4px or 8px base unit** consistently — never arbitrary values like 13px or 22px.
- **Whitespace is structure** — generous negative space communicates confidence. Dense layouts communicate utility tools.
- Avoid centering everything. Centered layouts feel like landing pages. Left-aligned or grid-broken layouts feel like products.
- **Asymmetry creates tension and interest.** A perfectly symmetric layout is often the laziest choice.

### Visual Hierarchy
- Every screen needs exactly one **focal point** — the element with the highest visual weight.
- Supporting elements should recede. Use opacity (0.4–0.6), smaller size, or lower contrast — not absence.
- **Borders are not free.** Every border adds visual noise. Use spacing and background color differences to separate regions first.
- Cards should not be nested inside cards. Depth should be earned.

---

## Anti-Pattern Library

These are patterns the model should actively avoid:

| Anti-Pattern | Why It's Bad | Better Alternative |
|---|---|---|
| Purple-to-blue gradient hero | Overused, meaningless | Flat bold color or editorial photo treatment |
| Inter + gray body text | Default, forgettable | Distinctive display font, high-contrast text |
| Three-column feature cards | Template-feel | Asymmetric grid, varying card sizes |
| Centered everything | Generic landing page | Left-aligned or deliberate grid |
| Icon + title + paragraph card grid | Zero visual interest | Integrated text-image layouts |
| `border-radius: 8px` everywhere | Indiscriminate rounding | Deliberate radius choices per element type |
| Box shadows on every card | Visual noise | Separate with spacing/color, not shadow |
| Emoji as iconography | Unprofessional | Lucide, Phosphor, or custom SVG icons |
| `opacity: 0.5` gray text | Low contrast, lazy | Proper color tokens for secondary text |
| Full-width hero with stock photo | Generic | Abstract, typographic, or product-first heroes |

---

## Command Vocabulary

Use these directives when working with the AI:

| Command | What It Does |
|---|---|
| `/impeccable init` | Set up `DESIGN.md` and `PRODUCT.md` for project context |
| `/impeccable audit` | Run accessibility and performance quality checks |
| `/impeccable critique` | UX design review — hierarchy, clarity, flow |
| `/impeccable polish` | Final refinement pass — spacing, typography, micro-details |
| `/impeccable distill` | Strip to essence — remove visual noise, clarify signal |
| `/impeccable animate` | Add purposeful, intentional motion |
| `/impeccable recolor` | Revisit color system — apply OKLCH, fix token consistency |
| `/impeccable retype` | Audit and improve typographic hierarchy |
| `/impeccable layout` | Revisit spatial structure and grid usage |
| `/impeccable contrast` | Accessibility contrast audit (WCAG AA/AAA) |

---

## Technical Constraints

- **CSS Custom Properties (variables)** for all tokens. No hardcoded values in component styles.
- **OKLCH for colors**: `oklch(65% 0.18 240)` not `#3b82f6`.
- **`clamp()` for fluid typography**: `font-size: clamp(1rem, 2.5vw, 1.5rem)`.
- **`dvh` for viewport heights**: `min-height: 100dvh` not `100vh` — prevents mobile browser chrome issues.
- **Container queries** over media queries where the component's container matters more than the viewport.
- **Logical properties**: `margin-inline`, `padding-block` for internationalization support.
- **`@layer`** for CSS cascade organization: reset → tokens → base → components → utilities.

---

## Motion & Animation

- Animate only `transform` and `opacity` — no layout-triggering properties.
- Entry animations: `ease-out`, 200–350ms.
- Exit animations: `ease-in`, 150–200ms (faster than entry).
- **Purposeful motion only** — every animation should communicate state change or guide attention.
- Respect `prefers-reduced-motion` — disable or minimize all animations when active.
- No decorative looping animations on production interfaces.

---

## Accessibility Baseline

- **WCAG AA minimum** for all text (4.5:1 for body, 3:1 for large text).
- All interactive elements keyboard accessible with visible focus states.
- `aria-label` or `aria-labelledby` on all icon-only buttons.
- Color is never the sole differentiator for state (e.g., error states must also use text or icon).
- Skip navigation link at page top.

---

## Quality Checklist

Before delivering any UI output:

- [ ] Is there a clear focal point on every screen?
- [ ] Is the typographic scale meaningful and consistent?
- [ ] Are colors from a defined token system (not ad-hoc)?
- [ ] Is spacing on a 4px or 8px grid?
- [ ] Are anti-patterns from the library avoided?
- [ ] Does the design work in both light and dark modes?
- [ ] Is `prefers-reduced-motion` respected?
- [ ] Does the layout work at 320px and 1440px viewport widths?
- [ ] Are all interactive elements accessible by keyboard?
- [ ] Does every visual decision serve a clear design purpose?
