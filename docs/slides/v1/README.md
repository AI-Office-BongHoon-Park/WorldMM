# V1 slides — superseded by V2

These are the first-cut single-case slide mockups (one HTML + one matching
`.slide.json` per case). They were the data-rich layout: every retrieved
spatial triple listed, every choice spelled out, both predictions shown side
by side.

**They are kept for traceability only.** The active deck lives one directory
up at `docs/slides/*-v2.html` and is regenerated from
`tools/render_v2_slides.py`.

The V1 → V2 redesign was driven by reviewer feedback that V1 did not make
the *spatial-helps-vs-baseline* contrast visceral within two seconds. V2
keeps the same factual content but renders a **left = without spatial /
center = "+ one fact" connector / right = with spatial** layout, with the
deciding triple promoted to a glowing "smoking gun" card and the final
answer letter at 46 px.

## Mapping

| V1 file | V2 file |
|---|---|
| `q1-screwdriver.html` | `../q1-screwdriver-v2.html` (hand-built reference) |
| `q33-takeout-app.html` | `../q33-takeout-app-v2.html` |
| `q53-hot-pot.html` | `../q53-hot-pot-v2.html` |
| `p6-meeting-room-floor.html` | `../p6-meeting-room-floor-v2.html` |
| `n1-kitchen-proximity.html` | `../n1-kitchen-proximity-v2.html` |
| `c1-kitchen-contains.html` | `../c1-kitchen-contains-v2.html` |
| `q6-grow-flowers.html` | `../q6-grow-flowers-v2.html` |
| `q45-coffee-timing.html` | `../q45-coffee-timing-v2.html` |
| `q50-phone-habit.html` | `../q50-phone-habit-v2.html` |
| — | `../new-where-am-i-v2.html` (V2-only authored scenario) |
| — | `../new-group-locator-v2.html` (V2-only authored scenario) |
| — | `../new-left-of-direction-v2.html` (V2-only authored scenario) |

## JSON specs

The V1 `.slide.json` files are preserved here for any downstream tool that
already parses them; the V2 templater (`tools/render_v2_slides.py`) keeps
the same per-case data as inline Python `Case` dataclasses, so a future
sync script can regenerate either set from a single source of truth if
needed.
