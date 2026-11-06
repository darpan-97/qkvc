// The builder's base64 model, in JavaScript, on the real weights (weights.json, written by export.py).
// The same arithmetic as qkvc/run.py's steps(): the stream is written group by group, each group's
// inputs then its outputs, and attention sees the last W tokens. Each output: embed -> 5 attention
// heads -> the MLP -> the output layer -> the highest score -> append it -> again.
// check_engine.js holds it equal to the Python model on known answers, and to base64 itself.
//
// Every step returns a trace of what was computed, for the page to show: where each head landed and
// how far back that is (with its weight and the runner-up's score), what it brought, which MLP units
// fired, and the scores of the output layer.

class Base64Transformer {
  constructor(w) {
    this.w = w
    this.id = Object.fromEntries(w.tokens.map((t, i) => [t, i]))
    const rows = (byRow) => {                       // {row: [[col, v]]} -> array indexed by row
      const out = new Array(w.size)
      for (const r in byRow) out[r] = byRow[r]
      return out
    }
    this.table = rows(w.token_table)
    this.heads = w.heads.map((h) => ({ name: h.name, Q: rows(h.Q), K: rows(h.K), V: rows(h.V), O: h.O,
                                       valueSize: h.value_size }))
    this.mlp = { W_in: rows(w.mlp.W_in), b_in: w.mlp.b_in, W_out: w.mlp.W_out, units: w.mlp.units }
    this.unitInputs = w.mlp.units.map(() => [])     // per unit: [[stream index, weight]], for the cards
    for (const r in w.mlp.W_in) for (const [u, v] of w.mlp.W_in[r]) this.unitInputs[u].push([+r, v])
    this.U = rows(w.unembed.U)
    this.names = []                                  // a name for every number of the stream
    for (const [name, [start, width]] of Object.entries(w.blocks))
      for (let i = 0; i < width; i++) this.names[start + i] = width === 1 ? name : `${name}[${i}]`
  }

  at(name, i = 0) { return this.w.blocks[name][0] + i }

  // ---- arithmetic on sparse vectors: Map(index -> value), only non-zero entries ----------------
  times(x, rows, size) {                             // x . W, W given row by row
    const out = size ? new Float64Array(size) : new Map()
    for (const [i, xv] of x) {
      const row = rows[i]
      if (!row) continue
      for (const [col, v] of row) {
        if (size) out[col] += xv * v
        else out.set(col, (out.get(col) || 0) + xv * v)
      }
    }
    return out
  }

  dot(a, b) {
    let s = 0
    for (const [i, v] of a) { const u = b.get(i); if (u) s += v * u }
    return s
  }

  symbols(mode, input) {                             // the task's read(): the input as token names
    return mode === 'decode' ? [...input] : [...input].map((b) => 'x' + b.toString(16).padStart(2, '0'))
  }

  // ---- a run: start() once, then step() for every output token --------------------------------
  // ids: every token of the stream so far. slots: what each one is, {kind, group, i}.
  start(mode, input) {
    const m = this.w.modes[mode]
    this.run = { mode, m, input, body: this.symbols(mode, input), ids: [], slots: [], outputs: [], done: false,
                 group: 0, left: 0, cache: this.heads.map(() => ({ K: [], V: [] })), row: null }
    return this.run
  }

  push(token, slot) {                                // embed one token and keep the last W keys and values
    const run = this.run, x = new Map()
    for (const [col, v] of this.table[token]) x.set(col, v)
    x.set(this.at('phase', run.ids.length % this.w.phases), 1)
    this.heads.forEach((h, i) => {
      const c = run.cache[i]
      c.K.push(this.times(x, h.K))
      c.V.push(this.times(x, h.V, h.valueSize))
      if (c.K.length > this.w.window) { c.K.shift(); c.V.shift() }
    })
    run.ids.push(token)
    run.slots.push(slot)
    run.row = x
  }

  feed() {                                           // the next group's mode token and inputs
    const run = this.run, [sizeIn, sizeOut] = run.m.group, g = run.group
    this.push(this.id[run.m.flag.toUpperCase()], { kind: 'mode', group: g })
    for (let i = 0; i < sizeIn; i++) {
      const at = g * sizeIn + i
      if (at < run.body.length) this.push(this.id[run.body[at]], { kind: 'in', group: g, i: at })
      else this.push(this.id.NONE, { kind: 'none', group: g })
    }
    run.group++
    run.left = sizeOut
  }

  step() {
    const run = this.run
    if (run.done) return null
    if (run.left === 0) this.feed()
    const x0 = run.row, x = new Map(x0), query = run.ids.length - 1
    const first = query - run.cache[0].K.length + 1  // the stream index of the window's oldest token
    const heads = this.heads.map((h, i) => {
      const q = this.times(x0, h.Q)
      const { K, V } = run.cache[i]
      const scores = K.map((k) => this.w.sharp * this.dot(q, k))
      const top = Math.max(...scores)
      const e = scores.map((s) => Math.exp(s - top))
      const total = e.reduce((a, b) => a + b, 0)
      const weights = e.map((v) => v / total)
      const value = new Float64Array(h.valueSize)
      weights.forEach((wt, j) => { if (wt) for (let d = 0; d < h.valueSize; d++) value[d] += wt * V[j][d] })
      const added = this.times(new Map([...value.entries()].filter(([, v]) => v)), h.O)
      for (const [col, v] of added) x.set(col, (x.get(col) || 0) + v)
      let top1 = 0
      weights.forEach((wt, j) => { if (wt > weights[top1]) top1 = j })
      const rival = scores.reduce((best, s, j) => (j !== top1 && s > best.s ? { s, j } : best), { s: -Infinity, j: -1 })
      return { name: h.name, pick: first + top1, back: query - first - top1, weight: weights[top1], score: scores[top1],
               rival, value: [...value], first, scores, weights }
    })
    // the MLP: ReLU(x . W_in + b) . W_out, added to the stream
    const pre = this.times(x, this.mlp.W_in, this.mlp.b_in.length)
    const hidden = pre.map((v, u) => Math.max(0, v + this.mlp.b_in[u]))
    for (const [u, row] of Object.entries(this.mlp.W_out)) {
      if (!hidden[u]) continue
      for (const [col, v] of row) x.set(col, (x.get(col) || 0) + hidden[u] * v)
    }
    // the output layer: one score per token
    const scores = this.times(x, this.U, this.w.tokens.length)
    scores.forEach((_, t) => { scores[t] += this.w.unembed.c[t] })
    let token = 0
    scores.forEach((s, t) => { if (s > scores[token]) token = t })
    const ranked = [...scores.keys()].sort((a, b) => scores[b] - scores[a])
    const n = run.outputs.length
    const trace = { n, token, name: this.w.tokens[token], scores: [...scores], ranked,
                    gap: scores[ranked[0]] - scores[ranked[1]], heads, hidden: [...hidden],
                    pre: [...pre].map((v, u) => v + this.mlp.b_in[u]), stream: x, queryRow: x0, query }
    if (this.w.tokens[token] === 'END') {
      run.done = true
    } else {
      run.outputs.push(token)
      run.left--
      this.push(token, { kind: 'out', group: run.group - 1, i: n })
    }
    return trace
  }

  // the whole answer, as text (encode) or bytes (decode)
  answer(mode, input) {
    this.start(mode, input)
    while (this.step());
    const names = this.run.outputs.map((t) => this.w.tokens[t])
    return mode === 'decode' ? Uint8Array.from(names.map((s) => parseInt(s.slice(1), 16))) : names.join('')
  }
}

if (typeof module !== 'undefined') module.exports = { Base64Transformer }
