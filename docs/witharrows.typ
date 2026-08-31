/// by Ulrik Sverdrup "bluss", dual licensed MIT/Apache-2.0

#let _fold-stroke(outer, inner) = if inner == auto { outer } else {
  let inner = stroke(inner)
  // Copied from elembic with thanks to Pg.
  stroke(
    paint: if inner.paint == auto { outer.paint } else { inner.paint },
    thickness: if inner.thickness == auto { outer.thickness } else { inner.thickness },
    cap: if inner.cap == auto { outer.cap } else { inner.cap },
    join: if inner.join == auto { outer.join } else { inner.join },
    dash: if inner.dash == auto { outer.dash } else { inner.dash },
    miter-limit: if inner.miter-limit == auto { outer.miter-limit } else { inner.miter-limit },
  )
}

/// round length in unit pt to enough precision
#let round-pt(x) = calc.round(x.pt(), digits: 4) * 1pt

#let consecutive-group-by(data, key: x => x) = {
  let group = ()
  let groups = ()
  let k = none
  for elt in data {
    let thiskey = key(elt)
    if k != thiskey {
      if group != () { groups.push(group) }
      group = ()
      k = none
    }
    if k == none {
      k = thiskey
    }
    group.push(elt)
  }
  if group != () { groups.push(group) }
  groups
}

#let clamplimit(value, limit) = {
  calc.max(-limit, calc.min(value, limit))
}

#let walabel(name) = label("_witharrows_" + name)


/// A "with arrows" environmet
/// Adding arrows with explanations between lines.
/// The equation must be multiline and every line, including the last one, must
/// be terminated with a linebreak (backslash).
///
/// Explanation arrows are inserted using the `explain` function.
///
/// The `equate` package is supported (it redraws equations in a grid, which is
/// actually a quite big change.), but it requires that every line
/// is marked with an explanation or with `explain(#none)`, so that the
/// witharrows can look up the position of each end of line.
///
/// by Ulrik Sverdrup "bluss", dual licensed MIT/Apache-2.0
#let witharrows(eq, pad-eq: 1em, pad-arrow: 1em, stroke: 0.06em, arrowhead-scale: 1) = {
  let stroke = std.stroke(stroke)
  show linebreak: it => [#metadata(none)#walabel("linebreak")] + it
  context {
    let parts = query(
      selector.or(walabel("linebreak"), walabel("explain"))
        .after(here())
        .before(selector(walabel("end")).after(here())),
    )
    let lines = ()
    let explanations = ()
    for part in parts {

      // part is either a marked linebreak or an explanation
      if part.label == walabel("linebreak") {
        lines.push(part.location().position())
      } else if part.label == walabel("explain") {
        explanations.push((
          pos: part.location().position(),
          ..part.value,
        ))
      } else {
        panic("unknown part: " + repr(part))
      }
    }
    let linegrid = (
      consecutive-group-by((explanations.map(elt => elt.pos) + lines)
        .sorted(key: elt => elt.y), key: elt => round-pt(elt.y))
        .map(line => {
          (min-x: round-pt(line.at(0).x), max-x: round-pt(line.at(-1).x), y: round-pt(line.at(0).y))
        })
    )
    let min-x = calc.min(calc.inf * 1pt, ..linegrid.map(elt => elt.min-x))
    let max-x = calc.max(0pt, ..linegrid.map(elt => elt.max-x))
    let cancel-math-space = h(0pt, weak: true)

    show walabel("explain"): it => {
      if it.value.expr == none { return it }
      cancel-math-space
      // The below does not work with dir: rtl, so until that's fixed, assume/use ltr
      set text(dir: ltr)
      let pos = it.location().position()
      let pos = (x: round-pt(pos.x), y: round-pt(pos.y))
      let expl = it.value
      let span = expl.span
      let updw = if span >= 0 { 1 } else { -1 }  // up or down
      let lfrt = if expl.side == end { 1 } else { -1 } // left or right side

      // find the next line
      let next-line = if updw >= 1 {
        linegrid.filter(elt => elt.y > pos.y).at(span - 1, default: none)
      } else {
        linegrid.filter(elt => elt.y < pos.y).rev().at(-span - 1, default: none)
      }
      let endy = if next-line != none { next-line.y } else { pos.y }
      let starty = pos.y
      let linediff = endy - pos.y
      let lineheight = 1em

      let pad-extra = if expl.place and lfrt > 0 { max-x - pos.x } else { 0pt }

      let columns = (pad-eq + pad-extra, pad-arrow, auto)
      let contents = (
        none,
        {
          let curve-pad = 0.15em * updw
          let curve-len = endy - starty - curve-pad * 2
          let cont-y = curve-len * 0.3 // control point lengths
          let cont-x = curve-len * 0.2
          let cont-x = clamplimit(cont-x.to-absolute(), 0.72em.to-absolute()) // limit width of the arc
          let cont-y = clamplimit(cont-y.to-absolute(), 0.92em.to-absolute())
          let arrow-unit = 0.12em * arrowhead-scale

          // Draw arrow in pure typst to simplify
          place(expl.side.inv(), dy: curve-pad + lineheight / 4, {
            curve(
              stroke: _fold-stroke(stroke, expl.stroke),
              curve.cubic((updw * cont-x * lfrt, cont-y), (updw * cont-x * lfrt, curve-len - cont-y), (0pt, curve-len)),
              curve.move((2 * arrow-unit * lfrt, curve-len - updw * arrow-unit)),
              curve.line((0pt, curve-len)),
              curve.line((0pt, curve-len - updw * arrow-unit * 2.15)),
            )
          })

        },
        move(dy: (endy - starty)/2 - lineheight / 3, {
          math.equation(it.value.expr)
        }),
      )
      if expl.side != end {
        columns = columns.rev()
        contents = contents.rev()
      }

      let maybe-place = if expl.place {
        let align = expl.side.inv()
        let dx = if expl.side != end { min-x - pos.x } else { 0pt }
        place.with(align, dy: - lineheight/3, dx: dx)
      } else { x => x }

      maybe-place(grid(
        // stroke: 0.2pt + blue,
        columns: columns,
        ..contents
      ))
      cancel-math-space
    }
    eq
  }
  [#metadata(none)#walabel("end")]
}


/// Explain the step from this equation line to the next
///
/// - expr (content, none): The explanation
/// - span (int): the span of the arrow in number of equation lines
/// it should traverse (which next line it should point to)
/// by default (1), the next line. The span can also be negative to point
/// to a prior line.
/// - stroke (length, color, stroke): override stroke of this arrow
/// - style (function, none): style function for the explanation
/// - place (bool): if true, use place for the explanation, so that it
/// does not affect the layout of the equation itself.
#let explain(expr, span: 1, stroke: auto, style: none, side: end, place: false) = {
  assert(side == start or side == end)
  if expr != none and style != none { expr = style(expr) }
  [#metadata((expr: expr, span: span, stroke: stroke, side: side, place: place))#walabel("explain")]
}