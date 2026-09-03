# Launch page

A single self-contained HTML file. Open it in any browser — no build step, no
server, no dependencies.

```
site/index.html
```

## What it does

Scroll position drives flight state. One scroll value feeds the sky gradient,
the telemetry rail, the phase callouts and the vehicle, so the page climbs
through a launch as you read it: hold, ignition, liftoff, max-Q, stage
separation, orbital insertion. Below insertion, the payload section explains
the project.

The altitude and velocity readouts follow physically shaped curves rather than
scroll percentage, landing near a real sun-synchronous insertion (~720 km,
~7.5 km/s). The range clock holds T-minus until the vehicle leaves the pad.

## Changing things

| What | Where |
|---|---|
| Project name | Search `DrishtiX` — appears in the title, badge and footer |
| Colours | The `:root` block at the top of the `<style>` |
| Phase copy | The five `.callout` blocks |
| Numbers | The `.readouts` blocks — keep these matching `eval/reports/` |
| Vehicle markings | The `#markings` group in the SVG — see note below |
| Launch complex | `#treeline`, `#mst`, `#masts`, `#lagoon` — all register to a 19vh horizon |
| Flight profile | The `flight()` function in the script |
| Phase timing | The `callouts` array and the thresholds in `flight()` |

If you rename the project, update `README.md` and `pyproject.toml` too.

## The launch complex

Built from the real site rather than generic architecture. Sriharikota is a
low barrier island — about 1 m elevation — between the Pulicat lagoon and the
Bay of Bengal, and it was a eucalyptus and casuarina firewood plantation
before it became a spaceport. So: a dense casuarina treeline on the horizon,
the lagoon glinting to the west, dawn breaking to the east where launches fly
out over the sea, lightning arrestor masts flanking the pad, and floodlit
concrete apron in the foreground.

The signature is the **Mobile Service Tower**. On a real pad it rolls back on
rails during the count, so here it retreats as you scroll — enclosing the
vehicle at T−10, clear of it by ignition. The background participates in the
launch instead of sitting behind it.

Tree positions are generated from a fixed seed, so the silhouette is
identical on every load. Regenerate by editing the `<use>` list in
`#treeline`.

## Two surfaces

The page changes material halfway down, and that is the point.

**Flight is dark** — you are looking up through the sky, so the background
gradient doubles as the altimeter.

**Payload is paper** — past orbital insertion you are looking *down* at Earth
data, so the page becomes what that work is done on: a topographic sheet with
contour lines, ink type and survey green. Orange survives from the flight but
is now reserved for exactly one thing — evidence.

The first version of this half was a dark slab with a hairline grid of
numbered `01–06` cells. That is the default look, it described the system in
prose instead of showing it, and it was correctly called stale. It was
replaced with the console.

## The console

`#console` runs the product rather than describing it. Three scenarios cycle:
a grounding query, a coverage query, and a **refusal** — which is deliberately
in the rotation rather than buried, because a system declining cleanly is the
most persuasive thing it does.

The plan steps light only when they actually run. On the refusal the sequence
halts at `validate` and the remaining four stay greyed, with
`refused at validate · no model run` underneath.

Edit `SCENARIOS` in the console script to change the questions. It only
animates while on screen, and a generation token retires a loop when you
scroll away so two cycles can never write to the DOM at once.

## About the vehicle markings

The `#markings` group carries the national flag as flown on a launch vehicle,
plus a mission roundel drawn for this project. **The roundel is ours, not an
official organisational emblem** — no real agency logo is reproduced here. If
you want to use official artwork, replace that group; it is fenced with
`MISSION MARKINGS` comments so it is easy to find. Check the usage terms for
whatever you drop in.

## Notes

- Webfonts load from Google Fonts but every stack has a system fallback, so
  the page still typesets correctly with the network off. Demo laptops are
  usually offline.
- `prefers-reduced-motion` is respected: the scroll animation is replaced by a
  composed static frame with the copy in normal document flow.
- Responsive to 390px. On mobile the telemetry becomes a strip along the
  bottom and the copy left-aligns.
- No canvas libraries, no scroll frameworks. One `requestAnimationFrame` loop
  writing only transforms and opacity.
