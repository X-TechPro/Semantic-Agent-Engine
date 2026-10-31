DESIGN LANGUAGE: "Terminal-technical dark console"
Applies to dashboards, settings, billing, profile, docs-style pages, and marketing pages.

Follow literally. Do not add rounded cards, gradients, shadows, glass/blur, emoji, stock illustrations, or decorative color.

═══ 1. CORE IDEA ═══
Pure-black canvas. Structure comes only from 1px hairlines, dashed hairlines, and tiny square "registration marks" at panel corners. Three type voices: a blocky pixel wordmark, a neutral grotesk for readable text, and monospace uppercase micro-labels for everything that is metadata. Data is shown with big bold numerals and segmented LED-style meters. Chrome is monochrome. Color appears only in data, status, and price highlights.

═══ 2. TOKENS ═══
:root {
  --bg:            #000;      /* page AND panels. Panels have no fill */
  --bg-sunken:     #0a0a0a;   /* group rows, heatmap well */
  --bg-control:    #141414;   /* filled buttons, ID chips, step squares */
  --bg-active:     #262626;   /* active nav item */
  --border:        #1f1f1f;   /* default hairline */
  --border-strong: #333;      /* buttons, tags, panels needing emphasis */
  --border-dashed: #2a2a2a;   /* dashed dividers/empty states, 1px dashed */
  --text:          #fff;
  --text-muted:    #a3a3a3;   /* body, nav, descriptions */
  --text-dim:      #6b6b6b;   /* mono labels, captions, "//" */
  --seg-off:       #161616;   /* empty meter segments */

  /* Data accents (use sparingly, never as large fills except inside data) */
  --mint:   #5eead4;   /* limit meters, tool bars, heatmap highlights */
  --purple: #8b6ff0;   /* token charts, model bars, bullets */
  --peach:  #f4b183;   /* run-count chart */
  --blue:   #6c7bff;   /* price highlights, promo tags, links in pricing ("$10 in credits included", "Pay as you go") */
  --green:  #22c55e;   /* success status text */
  --green-bg:#052e16;  /* success status fill */
  --green-bd:#15803d;  /* success status border */
  --amber:  #f59e0b;   /* "coming soon", warning, streak-zero tag */
  --focus:  #3b82f6;   /* keyboard focus ring: 2px solid, offset 0 */
}

═══ 3. TYPOGRAPHY ═══
a) Wordmark: heavy, condensed, pixel/stencil blocky uppercase, white. Logo only (substitute Bebas Neue / Anton / Big Shoulders Display Black, or a custom SVG).
b) Sans (Inter, Geist, or Helvetica Neue): nav 14px/500; page/section titles 20-24px/600 with tight tracking; body 13-14px, --text-muted; table body cells 13-14px, --text (tabular-nums); small descriptions 12px.
c) Mono (JetBrains Mono, Geist Mono, or IBM Plex Mono):
   - Micro-label: 10-12px, UPPERCASE, letter-spacing .08-.12em, --text-dim.
   - Hero number: 48-56px/700 white (dashboard stats). Price numbers 26-40px/700 in SANS.
   - Inline meta: usernames, IDs, timestamps, captions, "Showing 10 items", "resets Nov 24" (12px, --text-dim).
Section header pattern: "//" (mono, italic or regular, --text-dim) + title. Title is either bold sans ("// Overview", "// Plans") or mono UPPERCASE with optional 12px outline icon between ("// ▣ PINNED TASTE PACKAGES"). Eyebrow variant above an H1: "// PROVIDER API" in mono uppercase dim, then the H1 in sans 24px/600.

═══ 4. LAYOUT ═══
- App shell: left sidebar ~250px, pure black, 1px right border. Logo top-left (~160px wide, 24px top padding). Nav list below. Social icon row (18px, --text-dim) pinned to bottom.
- Optional top bar ~80px with bottom hairline: 16px outline icon + page title (16px/500) left, 32px circular avatar (initial letter, slate-gray fill) right. Some pages (profile) omit it.
- Content padding 24px. Sections separated by 48px. Marketing/settings content may sit in a centered column ~770px wide inside the content area.
- Two-column section layout is common: narrow left title/description column (~280px), wide right content.

═══ 5. SIGNATURE PRIMITIVES ═══
Panel: transparent background, 1px --border, radius 0, padding 24px. Optional header row separated by a 1px hairline (header = mono label row, ~56px tall, padding 0 16px).
Corner markers: 5px solid squares (--border-strong) centered on each corner of important panels/tables.
  .panel{position:relative;border:1px solid var(--border)}
  .panel::before,.panel::after{content:"";position:absolute;width:5px;height:5px;background:#444;left:-3px;top:-3px;box-shadow:calc(100% + 0px) 0 0 #444}
  (simpler: four absolutely positioned 5px spans)
Dashed variants: empty states and inner dividers use 1px dashed --border-dashed (e.g. "No pinned taste packages yet": centered outline icon + 13px muted text, ~125px tall).
Segmented meter: row of 22-30 equal squares, gap 2-3px, height 18px (8px in table rows), radius 0. On = --mint (or --purple in tables), off = --seg-off. Label left (mono uppercase), percentage right (mono bold 16px white), caption below (mono dim).
Thin progress bar (billing): 6px track --seg-off, fill #fff (white, not accent), radius 0, caption below in mono 12px: "31% of monthly limit used · resets Nov 24".
Mini bar chart: ~100px tall, 8-10px gaps, flat tops, one accent color, empty days = baseline hairline only, axis labels at both ends in mono uppercase dim ("OCT 24" … "NOV 24").
Heatmap (activity): tiny horizontal tick-bars in teal at 3 intensities, on --bg-sunken well, framed with teal L-shaped corner brackets (12px arms, 1px), legend "HIGH / MED / LOW" mono 10px dim with small squares.

═══ 6. COMPONENTS ═══
Nav item: 44px tall, 16px outline icon + 14px/500 label, gap 14px, --text-muted. Active: bg --bg-active, white text, radius 2-4px. Focus: 2px --focus outline around the item.
Button (default): transparent, 1px --border-strong, radius 2px, 34px tall, padding 0 16px, 13px/500, white text, optional 14px leading icon. Hover: border #666.
Button (filled): bg --bg-control, same border (e.g. "Edit profile" full width, "Generate API key").
Button (ghost): no border, --text-muted, chevron icon; disabled state dimmer ("‹ Newer"). Paired primary uses normal button ("Older ›").
Text-link action: 12px muted + "↗" glyph ("Manage API keys ↗", "How Go usage limits work ↗"); underline on hover. Inline links in descriptions are underlined.
Pill/tag (label): mono 10-11px uppercase, 1px border, radius 0-2px, padding 2px 8px. Variants:
  - neutral: --border-strong, --text-dim (e.g. "52w", "CURRENT BILLING MONTH")
  - NEW: white fill, black text, bold
  - status success: mono uppercase text --green, fill --green-bg, 1px --green-bd, fixed width ~144px, centered ("COMPLETED"); active badge uses small green dot + "ACTIVE"
  - warning: amber dot + mono uppercase text inside bordered chip ("● COMING SOON"); amber tag for "0d STREAK"
  - promo: --blue text on dark-blue fill (#12143a), 1px blue border ("* All deals apply")
  - DEAL tag: dashed border, mono 10px, dim, letter-spaced
ID chip: --bg-control fill, mono 12px dim, truncated with "…" ("6d0dff8c …"), radius 2px, padding 4px 10px.
Segmented toggle (plan tiers): row of 3 mono 10px labels spaced-out ("Pro $20 | Max 10× | Max 20×") inside a bordered strip; active segment = white fill, black text.
"CURRENT" ribbon: full-width white bar at top of a card, black mono bold uppercase text with "✦" prefix, letter-spacing .15em, 32px tall.
Select: 1px border, bg black, mono 12px, small chevron, ~64px wide.
Avatar (profile): 280px SQUARE (radius 0), solid slate (#4a6572), huge white sans initial centered. Header avatar is a 32px circle.

Table (dense data): outer 1px border + corner markers; header row = mono uppercase dim with optional small info icon; columns separated by 1px vertical lines in summary tables, none in log tables; row 46-51px, 1px bottom hairline; row cells sans 13-14px white, tabular numbers; numeric columns right- or left-aligned consistently; group rows on --bg-sunken with 8px colored square bullet + mono uppercase label + dim totals. Footer below table: mono dim "// Showing 10 items", "per page:" + select, right-aligned Newer/Older buttons.

Description list (feature rows): 2 columns separated by hairline rows: left bold 14px label (~176px column), right 12px muted copy, 20px vertical padding.
Vertical stepper: 32px square step markers (--bg-control, 1px border, white mono digit) connected by a 1px vertical line; to the right: title 14px/600 white, subtitle 12px muted, then the step's content (card/button).
Pricing card (2-column): left: mono-bold plan name (+ NEW tag), price 26px/700 sans + "/month" muted 13px + "+ processing fee" 10px dim, blue highlight text, muted description; bottom row full width with top hairline: "Upgrade →" (label left, arrow right, 44px tall, hover bg #0a0a0a). Right column: mono uppercase "WHAT'S INCLUDED" + list (icon 14px + 12px text, underlined links). Plan grids: 3 cards sharing one outer bordered frame with corner markers; each card: name (mono-ish 14px/600 uppercase), price 36px/700, "/month" muted, --blue line ("$10 in credits included"), muted usage line with bold figure and underlined "deals", one-line tagline in muted 12px, bottom CTA row, then "WHAT'S INCLUDED" list.
Deal row: [DEAL tag] + 13px text with inline `code` (mono) + right-aligned mono dim "while capacity lasts ↗"; rows separated by dashed hairlines.
Activity feed: rows with 14px icon, mono username + sans text, below a mono 10px uppercase dim timestamp ("3 DAYS AGO"); right-aligned mono 10px index "[01]" dim.
Locked/coming-soon overlay: underlying content at ~20% opacity on --bg-sunken, centered chip "● COMING SOON".
Icons: outline only, 1.5px stroke, round caps, 14-16px (Lucide / Phosphor Light). Brand logos (OpenAI, Anthropic, etc.) as monochrome gray glyphs, 20px, 20px gaps.

═══ 7. MOTION ═══
120-150ms ease for color/border hover only. Meter segments may fill left-to-right with 15ms stagger on mount. No bounce, parallax, scale, or glow.

═══ 8. HARD RULES ═══
1. Radius 0 everywhere except avatar circle; buttons and active nav max 2-4px.
2. No shadows, gradients, blur, or translucency except the "coming soon" dimmed content.
3. All micro-labels are mono, uppercase, letter-spaced, --text-dim. All hero numbers are big, bold, white.
4. Accent colors live in data, status chips, and price highlight lines only. Never color backgrounds of chrome, nav, or buttons.
5. One accent per component. Mint for limits/tools, purple for token/model share, peach for runs, blue for pricing/promos, green for success, amber for warnings.
6. Empty states use dashed borders. Active/important panels use solid borders with corner markers.
7. Copy is terse and technical: "Pick one. Switch anytime." / "Resets in 3h 52m" / "A buck and open models."
8. Dense but airy: generous panel padding, tight table rows, large gaps between sections.

═══ 9. BUILD INSTRUCTIONS ═══
When asked for a new page: choose a layout (sidebar app shell or centered column), express every piece of content using the primitives above (panels with hero numbers, segmented meters, hairline tables, dashed empty states, step lists, pricing grids), apply "// Title" headers to every section, and output production-ready HTML/CSS (or the framework requested) with the tokens defined as CSS variables.
