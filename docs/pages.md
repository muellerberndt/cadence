# A trained net in a browser page

A settled net is a matrix, a bias vector, and a rule. A page can hold all three and run
the settlement itself, so what you watch in the browser is the net, not a recording.

## Export

```python
engine = learner.engine
rule = engine.rule
payload = {
    "n": wiring.n,
    "sets": {name: list(members) for name, members in wiring.sets.items()},
    "W": engine.dense().ravel().round(5).tolist(),           # W[pre, post], row-major
    "bias": engine.bias.round(5).tolist(),
    "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak,
             "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
}
```

`Settlement.dense()` returns the overlap matrix with every effective drive folded in
(`gain`, `count`, `edge_scale`, `log_gain`), so the page needs no other number. A few
hundred owners is a few hundred kilobytes of JSON; embed it in the HTML.

## Settle

The whole engine, in the page:

```javascript
function act(v) {
  const r = 1 / (1 + Math.exp(-R.slope * (v - R.threshold))) - R.rest;
  return r > 0 ? r / (1 - R.rest) : R.leak * r / R.rest;
}
function settle(drive) {                       // drive: one number per owner, the clamp
  const v = new Float64Array(n), s = new Float64Array(n), inbox = new Float64Array(n);
  for (let t = 0; t < 100; t++) {
    inbox.fill(0);
    for (let i = 0; i < n; i++) { const si = s[i]; if (si === 0) continue;
      for (let j = 0; j < n; j++) inbox[j] += si * W[i * n + j]; }
    let moved = 0;
    for (let j = 0; j < n; j++) {
      v[j] += R.dt * (-v[j] + inbox[j] + drive[j] + BIAS[j]);
      const ns = act(v[j]); moved = Math.max(moved, Math.abs(ns - s[j])); s[j] = ns;
    }
    if (moved < 1e-4) break;
  }
  return s;
}
```

Same arithmetic as the NumPy backend, in the page's own precision. A 400-owner net
settles in a few milliseconds.

## Show the settlement

Record the output owners' activations at every step and replay them as bars over a few
animation frames: the user sees the answer form. Show the hidden owners at rest as a strip
of squares with opacity as activation. Print the number of steps to rest. Both game pages
in cadence-examples do exactly this, and their `page_template.html` files are a starting
point: `build_page.py` replaces one placeholder with the exported JSON.
