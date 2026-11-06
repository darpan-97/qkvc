// The page: runs the builder's base64 model (engine.js) one token at a time and shows each step's
// computation, left to right, in the manner of the Transformer Explainer: the window of the stream, the
// newest token's embedding, the 5 attention heads (how far back each looks, what it brings), the MLP's
// bit wiring, the output layer's scores, the token written. Every token is checked against the
// browser's own base64 (btoa / atob).

const $ = (s) => document.querySelector(s)
const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const FETCH = { decode: ['dec_first', 'dec_second'], encode: ['enc_first', 'enc_second'] }
const GROUP_IN = { decode: 4, encode: 3 }, GROUP_OUT = { decode: 3, encode: 4 }
let M, mode = 'encode', data = null, trace = null, expected = [], wrong = null, timer = null, tourAt = -1

// ---- input and the reference answer -----------------------------------------------------------
function readInput() {
  const text = $('#input').value
  if (mode === 'encode') return { ok: true, data: new TextEncoder().encode(text) }
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(text) || text.length % 4) return { ok: false, why: 'not standard base64 (length a multiple of 4, = only at the end)' }
  return { ok: true, data: text }
}

function reference(d) {           // the browser's own base64, as the model's token names
  if (mode === 'encode') return [...btoa(String.fromCharCode(...d))]
  return [...atob(d)].map((ch) => 'x' + ch.charCodeAt(0).toString(16).padStart(2, '0'))
}

function reset() {
  stop()
  const r = readInput()
  $('#input').classList.toggle('bad', !r.ok)
  if (!r.ok) { $('#status').innerHTML = `<span class="bad">${r.why}</span>`; return }
  data = r.data
  M.start(mode, data)
  M.feed()
  trace = null; wrong = null
  expected = reference(data)
  render()
}

function step() {
  if (!M.run || M.run.done || wrong !== null) return false
  trace = M.step()
  const want = trace.n < expected.length ? expected[trace.n] : 'END'
  if (trace.name !== want) wrong = { n: trace.n, got: trace.name, want }
  return !M.run.done && wrong === null
}

function stop() { clearInterval(timer); timer = null; $('#play').textContent = 'Play' }

// ---- names for tokens -------------------------------------------------------------------------
function show(name) {             // how a token is written on the page
  if (/^x[0-9a-f]{2}$/.test(name)) {
    const v = parseInt(name.slice(1), 16)
    return { main: name, small: v >= 33 && v < 127 ? `'${String.fromCharCode(v)}'` : v === 32 ? "' '" : '' }
  }
  if (name.length === 1 && ALPHABET.includes(name)) return { main: name, small: String(ALPHABET.indexOf(name)) }
  return { main: name, small: '' }
}
const tokName = (t) => M.w.tokens[t]
const bitsOf = (v, n) => [...Array(n)].map((_, k) => (v >> (n - 1 - k)) & 1).join('')
const short = (name) => {         // a token in the stream panel: x48 reads H, x20 reads _
  if (!/^x[0-9a-f]{2}$/.test(name)) return name
  const v = parseInt(name.slice(1), 16)
  return v === 32 ? '_' : v > 32 && v < 127 ? String.fromCharCode(v) : name
}
const distance = (place, f) => M.w.modes[mode].distances[place][f]   // how far back fetch f reads; null: not needed

// ---- render -----------------------------------------------------------------------------------
function render() {
  const run = M.run, ids = run.ids
  const n = trace ? trace.n : 0
  const place = n % GROUP_OUT[mode]
  // status line
  const total = expected.length
  const st = []
  st.push(!trace ? `ready: <b>${data.length}</b> input ${mode === 'encode' ? 'bytes' : 'characters'}, press Next token`
          : trace.name === 'END' ? `all <b>${total}</b> outputs written, then END`
          : `output <b>${n + 1}</b> of ${total}`)
  if (trace) st.push(`place <b>${place}</b> in its group of ${GROUP_OUT[mode]}`)
  if (trace) st.push(`wrote <b class="mono">${trace.name}</b>`)
  st.push(wrong ? `<span class="bad">output ${wrong.n} is ${wrong.got}, base64 says ${wrong.want}: stopped</span>`
                : `<span class="ok">matches the browser's own base64: all ${run.outputs.length + (run.done ? 1 : 0)} tokens</span>`)
  if (run.done && !wrong) st.push('<b>done</b>')
  $('#status').innerHTML = st.map((s) => `<span>${s}</span>`).join('')
  $('#next').disabled = $('#end').disabled = $('#play').disabled = run.done || wrong !== null

  const q = trace ? trace.query : ids.length - 1
  const heads = trace ? trace.heads : null
  const fetchIdx = FETCH[mode].map((nm) => M.heads.findIndex((h) => h.name === nm))
  const picks = heads ? fetchIdx.map((i, f) => (distance(place, f) === null ? undefined : heads[i].pick)) : []
  const fresh = trace && trace.name !== 'END' ? q + 1 : null

  // the window: the last W tokens, what attention can see, each with how far back it is
  const oldest = Math.max(0, q - M.w.window + 1)
  let html = oldest ? `<div class="gap">… ${oldest} earlier</div>` : ''
  for (let i = oldest; i < ids.length; i++) {
    const t = show(tokName(ids[i]))
    const cls = ['chip', run.slots[i].group % 2 ? 'g1' : '', i === q ? 'query' : '', i === picks[0] ? 'pick1' : '', i === picks[1] ? 'pick2' : '',
                 i === fresh ? 'new' : ''].join(' ')
    const where = i === fresh ? 'written' : i === q ? 'newest' : `${q - i} back`
    html += `<div class="${cls}" data-pos="${i}"><span>${t.main} <small>${t.small}</small></span><small>${where}</small></div>`
  }
  $('#seq').innerHTML = html
  const L = ids.length, mark = (i, colour) => `<i style="left:${(100 * i / Math.max(1, L - 1)).toFixed(2)}%;background:var(${colour})"></i>`
  $('#bar').innerHTML = (picks[0] !== undefined ? mark(picks[0], '--q') : '') + (picks[1] !== undefined ? mark(picks[1], '--k') : '') + mark(q, '--ink')
  $('#bar').title = `the whole sequence: ${L} tokens`

  // embedding of the newest token
  const row = trace ? trace.queryRow : M.run.row
  const feats = [...row.keys()].sort((a, b) => a - b)
  $('#emb').innerHTML = `<div class="note">${feats.length} of ${M.w.size.toLocaleString()} numbers are 1:</div>` +
    feats.map((i) => `<div class="feat">${M.names[i]}</div>`).join('') +
    `<div class="note" style="margin-top:6px">the rest are 0</div>`

  // attention
  $('#att').innerHTML = M.heads.map((h, i) => {
    if (!heads) return `<div class="head"><div class="hname">${h.name}</div><div class="q">?</div><div></div><div class="val">?</div></div>`
    const hd = heads[i]
    const off = h.name !== 'mode' && !FETCH[mode].includes(h.name)
    const pickedName = tokName(ids[hd.pick])
    const d = off || h.name === 'mode' ? null : distance(place, FETCH[mode].indexOf(h.name))
    let ask, val
    if (h.name === 'mode') { ask = 'the mode token'; val = `${pickedName} <small>mode_${mode === 'decode' ? 'dec' : 'enc'} = 1</small>` }
    else if (off) { ask = '—'; val = '—' }
    else if (d === null) { ask = 'not needed'; val = '—' }
    else {
      ask = `${d} back`
      const v = hd.value
      const bitsN = h.name.startsWith('dec') ? 6 : 8
      const num = v.slice(0, 8).reduce((s, b, k) => s + (Math.round(b) << k), 0)
      if (v[9] > 0.5) val = `missing <small>landed on ${pickedName}</small>`
      else if (v[8] > 0.5) val = `= <small>pad</small>`
      else { const t = show(pickedName); val = `<b>${t.main}</b> <small>${bitsOf(num, bitsN)}</small>` }
    }
    return `<div class="head${off ? ' off' : ''}" data-head="${i}"><div class="hname">${h.name}${off ? ' · other direction' : ''}</div>
      <div class="q">${ask}</div><div class="dot" style="opacity:${Math.max(0.15, hd.weight)}" title="weight ${hd.weight.toFixed(6)}"></div>
      <div class="val">${val}</div></div>`
  }).join('')

  renderWires(place)
  // output layer
  if (trace) {
    const top = trace.ranked.slice(0, 5)
    const best = trace.scores[top[0]]
    $('#out').innerHTML = top.map((t, k) => {
      const s = trace.scores[t], sh = show(tokName(t))
      return `<div class="score${k === 0 ? ' win' : ''}"><span>${sh.main}</span><div class="b" style="width:${Math.max(4, 100 * Math.max(0, s) / best)}%"></div><span>${s}</span></div>`
    }).join('') + `<div class="note">wins by ${trace.gap}; ${M.w.tokens.length - 5} more below</div>`
    const sh = show(trace.name)
    const ok = !wrong || wrong.n !== trace.n
    $('#next-tok').innerHTML = `<div class="next">${sh.main}</div><div class="check ${ok ? 'ok' : 'bad'}">${ok ? '✓ base64' : '✗ base64 says ' + wrong.want}</div>`
  } else {
    $('#out').innerHTML = '<div class="note">nothing yet</div>'
    $('#next-tok').innerHTML = ''
  }
  renderPanels(picks, place)
  requestAnimationFrame(drawRibbons)
}

// the MLP: this place's wires, from the two fetched symbols' bits to the output's bits
function renderWires(place) {
  const svg = $('#wires')
  const inBits = mode === 'decode' ? 6 : 8, outBits = mode === 'decode' ? 8 : 6
  const units = M.mlp.units.map((label, u) => ({ label, u, m: label.match(/^(\w+) place (\d): (first|second)\.bit(\d) -> out bit (\d)$/) }))
  const wires = units.filter((x) => x.m && x.m[1] === mode && +x.m[2] === place)
  const fetched = trace ? FETCH[mode].map((nm) => trace.heads[M.heads.findIndex((h) => h.name === nm)].value) : null
  const outVal = trace ? [...Array(8)].map((_, k) => trace.stream.get(M.at('out_bits', k)) || 0) : null
  const rowH = 18, top = 26
  const yIn = (which, k) => top + (which === 'first' ? 0 : inBits * rowH + 22) + (inBits - 1 - k) * rowH
  const outTop = top + ((2 * inBits * rowH + 22) - outBits * rowH) / 2
  const yOut = (o) => outTop + (outBits - 1 - o) * rowH
  let s = `<text class="lbl" x="6" y="14">first</text><text class="lbl" x="6" y="${top + inBits * rowH + 18}">second</text>` +
          `<text class="lbl" x="238" y="${outTop - 8}">output bits</text>`
  for (const w of wires) {
    const on = trace && trace.hidden[w.u] > 0.5
    const y1 = yIn(w.m[3], +w.m[4]) + 9, y2 = yOut(+w.m[5]) + 9
    s += `<path d="M64 ${y1} C 150 ${y1}, 150 ${y2}, 242 ${y2}" fill="none" stroke="var(${on ? '--ink' : '--rule'})" stroke-width="${on ? 1.8 : 1.2}"${on ? '' : ' stroke-dasharray="3 3"'}><title>${w.label}${trace ? ' · fired ' + (on ? '1' : '0') : ''}</title></path>`
  }
  const box = (x, y, v, colour, label, labelX) => `<rect x="${x}" y="${y}" width="22" height="16" rx="3" fill="var(${v ? colour + '-soft' : '--bg'})" stroke="var(${colour}-line)"/>` +
    `<text x="${x + 7}" y="${y + 12}">${v === null ? '·' : v}</text><text class="lbl" x="${labelX}" y="${y + 12}">${label}</text>`
  for (const [which, colour, f] of [['first', '--q', 0], ['second', '--k', 1]])
    for (let k = inBits - 1; k >= 0; k--)
      s += box(40, yIn(which, k), fetched ? Math.round(fetched[f][k]) : null, colour, k, 22)
  for (let o = outBits - 1; o >= 0; o--) s += box(244, yOut(o), outVal ? Math.round(outVal[o]) : null, '--code', 'bit ' + o, 270)
  svg.innerHTML = s
  $('#mlpsub').textContent = trace ? `place ${place}: ${wires.filter((w) => trace.hidden[w.u] > 0.5).length} of ${wires.length} wires carry a 1`
                                   : 'the fetched bits, rewired into the output\'s bits'
  const flags = units.filter((x) => !x.m && x.label.startsWith(mode))
  $('#flags').innerHTML = flags.map((x) => `<span class="flag${trace && trace.hidden[x.u] > 0.5 ? ' on' : ''}" title="${x.label}">${x.label.split(': ')[1]}</span>`).join('')
}

function renderPanels(picks, place) {
  $('#in-sub').textContent = `${M.run.ids.length} tokens so far, in groups of ${GROUP_IN[mode] + GROUP_OUT[mode] + 1}: mode, ${GROUP_IN[mode]} in, ${GROUP_OUT[mode]} out`
  $('#in-syms').innerHTML = M.run.ids.map((t, i) => {
    const cls = [M.run.slots[i].group % 2 ? 'g1' : '', i === picks[0] ? 'cur' : '', i === picks[1] ? 'cur2' : ''].join(' ')
    return `<span class="${cls}">${esc(short(tokName(t)))}</span>`
  }).join('')
  const outs = M.run.outputs.map(tokName)
  $('#out-sub').textContent = `${outs.length} of ${expected.length}`
  $('#out-syms').innerHTML = outs.map((s, i) => `<span class="${Math.floor(i / GROUP_OUT[mode]) % 2 ? 'g1' : ''}">${s}</span>`).join('') || '<span class="note">nothing yet</span>'
  const text = mode === 'encode' ? outs.join('') : new TextDecoder().decode(Uint8Array.from(outs.map((s) => parseInt(s.slice(1), 16))))
  $('#ref').innerHTML = M.run.done ? (mode === 'encode' ? `= <span class="mono">${esc(text)}</span>` : `as text: <span class="mono">${esc(text)}</span>`) : ''
  $('#rule').textContent = ruleText(place)
}

const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))

// the rule at this place, read off the MLP's wiring: which bits of which symbol land where
function ruleText(place) {
  const re = /^(\w+) place (\d): (first|second)\.bit(\d) -> out bit (\d)$/
  const wires = M.mlp.units.map((l) => l.match(re)).filter((m) => m && m[1] === mode && +m[2] === place)
  const terms = []
  for (const head of ['first', 'second']) {
    const ws = wires.filter((m) => m[3] === head).map((m) => [+m[4], +m[5]])
    if (!ws.length) continue
    const lo = Math.min(...ws.map((w) => w[0])), hi = Math.max(...ws.map((w) => w[0])), shift = ws[0][1] - ws[0][0]
    let t = lo ? `(${head} >> ${lo})` : head
    if (hi - lo + 1 < (mode === 'decode' ? 6 : 8) - lo) t = `(${t} & ${(1 << (hi - lo + 1)) - 1})`
    if (lo + shift) t = `${t} << ${lo + shift}`
    terms.push(t)
    terms.push(`   bits ${hi}..${lo} of ${head} -> output bits ${hi + shift}..${lo + shift}`)
  }
  const exprs = terms.filter((_, i) => i % 2 === 0), notes = terms.filter((_, i) => i % 2 === 1)
  return `${mode === 'decode' ? 'byte' : 'char'} = ${exprs.join(' | ')}\n${notes.join('\n')}`
}

// ---- ribbons: from each picked token to its head --------------------------------------------
function drawRibbons() {
  const svg = $('#ribbons'), box = $('#diagram').getBoundingClientRect()
  if (!trace) { svg.innerHTML = ''; return }
  const at = (el, side) => { const r = el.getBoundingClientRect(); return [side === 'r' ? r.right - box.left : r.left - box.left, r.top - box.top + r.height / 2] }
  let s = ''
  M.heads.forEach((h, i) => {
    const off = h.name !== 'mode' && !FETCH[mode].includes(h.name)
    if (off) return
    if (h.name !== 'mode' && distance(trace.n % GROUP_OUT[mode], FETCH[mode].indexOf(h.name)) === null) return
    const chip = document.querySelector(`.chip[data-pos="${trace.heads[i].pick}"]`)
    const q = document.querySelector(`.head[data-head="${i}"] .q`)
    if (!chip || !q) return
    const [x1, y1] = at(chip, 'r'), [x2, y2] = at(q, 'l')
    const colour = h.name === 'mode' ? '--ink-3' : FETCH[mode].indexOf(h.name) === 0 ? '--q' : '--k'
    const mx = (x1 + x2) / 2
    s += `<path d="M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}" fill="none" stroke="var(${colour})" stroke-width="2" opacity=".75"/>`
  })
  const qc = document.querySelector('.chip.query'), emb = $('#emb')
  if (qc && emb) {
    const [x1, y1] = at(qc, 'r'), [x2, y2] = at(emb, 'l')
    s += `<path d="M${x1} ${y1} C ${x1 + 20} ${y1}, ${x2 - 20} ${y2}, ${x2 - 2} ${y2}" fill="none" stroke="var(--ink)" stroke-dasharray="4 3" stroke-width="1.2"/>`
  }
  svg.innerHTML = s
}

// ---- + cards: the arithmetic of the step on screen -------------------------------------------
function card(html) { $('#card-body').innerHTML = html; $('#card').style.display = 'block' }
const DESCRIBE = {
  one: 'always 1, so a layer can add a constant', bits: 'the token\'s own value, one bit each', is_char: 'this token is a base64 character',
  is_byte: 'this token is a byte', is_pad: 'this token is =', is_none: 'this token is NONE: the group has run out of input',
  is_end: 'this token is END', is_dec: 'this token is DEC', is_enc: 'this token is ENC',
  phase: `where this token is: its index in the stream, counted from 0 and starting again at each group`,
}
function cardSeq() {
  const row = trace ? trace.queryRow : M.run.row
  const name = tokName(M.run.ids[trace ? trace.query : M.run.ids.length - 1])
  card(`<h4>The newest token, ${name}, as numbers</h4>
    <p class="note">Each token is one row of the embedding table (what it is) plus a 1 at its phase (where it is). Every group is
    ${M.w.phases} tokens, so the phase says where a token sits in its group, and it never runs out however long the input is.</p>
    <table><tr><th>number</th><th>value</th><th>what it says</th></tr>${[...row.keys()].sort((a, b) => a - b).map((i) =>
      `<tr><td>${M.names[i]}</td><td>${row.get(i)}</td><td>${DESCRIBE[M.names[i].split('[')[0]] || ''}</td></tr>`).join('')}</table>
    <p class="note">The other ${(M.w.size - row.size).toLocaleString()} numbers are 0.</p>`)
}
function cardHead(i) {
  if (!trace) return card('<h4>Attention</h4><p class="note">Press Next token first.</p>')
  const hd = trace.heads[i], h = M.heads[i], ids = M.run.ids
  const order = [...hd.scores.keys()].sort((a, b) => hd.scores[b] - hd.scores[a])
  const backOf = (j) => trace.query - hd.first - j
  const rows = order.slice(0, 3).map((j) => `<tr><td>${backOf(j)} back</td><td>${tokName(ids[hd.first + j])}</td><td>${hd.scores[j]}</td><td>${hd.weights[j].toExponential(2)}</td></tr>`).join('')
  const zero = hd.scores.filter((s) => s === 0).length
  const f = FETCH[mode].indexOf(h.name), place = trace.n % GROUP_OUT[mode], d = f < 0 ? null : distance(place, f)
  const phase = trace.query % M.w.phases
  card(`<h4>Head ${h.name}</h4>
    <p class="note">The query is made from the newest token's numbers. ${h.name === 'mode' ? 'It asks for the mode token (DEC or ENC).' :
      f < 0 ? 'This head belongs to the other direction: nothing reads what it brings.' :
      d === null ? `This output needs no ${h.name.split('_')[1]} symbol, so the query names no phase and the head falls back to the mode token, which reads as missing.` :
      `The newest token is at phase ${phase}, and the query weights turn that into <b>phase ${(phase - d + M.w.phases) % M.w.phases}</b>: the token <b>${d} back</b>, the same distance in every group.`}
    The key of each of the last ${M.w.window} tokens is compared with it, and the score is 100 × the match.</p>
    <table><tr><th>where</th><th>token</th><th>score</th><th>weight after softmax</th></tr>${rows}
    <tr><td colspan="2">${zero} other tokens</td><td>0</td><td>${zero ? Math.exp(-hd.score).toExponential(1) : ''} each</td></tr></table>
    <p class="note">weight = e<sup>score</sup> / Σ e<sup>score</sup>. The pick, ${hd.back} back, gets ${hd.weight.toFixed(15)} of the weight, so what the head brings
    is that token's numbers: <span class="mono">${hd.value.map((v) => Math.round(v * 1e6) / 1e6).join(' ')}</span>
    ${h.name === 'mode' ? '(mode_dec, mode_enc)' : '(bits 0-7, pad, missing)'}.</p>`)
}
function cardMlp() {
  if (!trace) return card('<h4>MLP</h4><p class="note">Press Next token first.</p>')
  const place = trace.n % GROUP_OUT[mode]
  const units = M.mlp.units.map((l, u) => ({ l, u })).filter((x) => x.l.startsWith(`${mode} place ${place}:`))
  const rows = units.map(({ l, u }) => {
    const ins = M.unitInputs[u].filter(([i]) => trace.stream.get(i)).map(([i, w]) => `${M.names[i]}=${+trace.stream.get(i).toFixed(6)}`)
    const pre = trace.pre[u]
    return `<tr><td>${l.split(': ')[1]}</td><td>${ins.join(' + ') || '—'}</td><td>${M.mlp.b_in[u]}</td><td>${+pre.toFixed(6)}</td><td><b>${+trace.hidden[u].toFixed(6)}</b></td></tr>`
  }).join('')
  card(`<h4>The MLP's units at place ${place}</h4>
    <p class="note">Each unit is an AND gate: ReLU(sum of its inputs + bias). Its inputs are this direction's mode flag, "this is place ${place}"
    (the phase of the token that predicts it), and one fetched bit. With bias −2 it gives 1 only when all three are 1. A unit that fires writes a 1
    into one output bit.</p>
    <table><tr><th>unit</th><th>inputs that are not 0</th><th>bias</th><th>sum</th><th>ReLU</th></tr>${rows}</table>
    <p class="note">The other ${M.mlp.units.length - units.length} units belong to other places or the other direction: their place or mode input is 0, so they give 0.</p>`)
}
function cardOut() {
  if (!trace) return card('<h4>Output layer</h4><p class="note">Press Next token first.</p>')
  const top = trace.ranked.slice(0, 6), x = trace.stream
  const rows = top.map((t) => {
    // a bit's part of the score: (the token's bit as +1/-1) x (the output bit as +1/-1); the weight
    // on out_bits[k] is 2s and the constant holds -s, so together they give s x (2 x out - 1)
    const bits = []
    for (let k = 7; k >= 0; k--) {
      const w = (M.U[M.at('out_bits', k)] || []).find(([col]) => col === t)
      if (w) bits.push([k, Math.sign(w[1]) * (2 * Math.round(x.get(M.at('out_bits', k)) || 0) - 1)])
    }
    const kind = trace.scores[t] - bits.reduce((s, [, v]) => s + v, 0)
    const bitCells = bits.map(([k, v]) => `<span class="pm ${v > 0 ? 'p' : 'm'}" title="bit ${k}">${v > 0 ? '+' : '−'}</span>`).join('')
    return `<tr><td>${tokName(t)}</td><td>${+kind.toFixed(6)}</td><td>${bitCells}</td><td><b>${trace.scores[t]}</b></td></tr>`
  }).join('')
  card(`<h4>The output layer: a score for each token</h4>
    <p class="note">Each candidate gets +1 for every output bit that matches its own bits and −1 for every one that does not (written as weights
    on the output bits plus a constant), and a large constant (20) that decides the kind: bytes when decoding, characters when encoding, =
    when padding, END (60) when stopping. The right token scores highest; the closest wrong one differs by one bit, so it is 2 lower.</p>
    <table><tr><th>token</th><th>kind part</th><th>bits, high to low (+ agrees)</th><th>score</th></tr>${rows}</table>`)
}

// ---- the tour ---------------------------------------------------------------------------------
const TOUR = [
  ['c-seq', () => `The stream is written group by group: the mode token, the group's inputs, then its outputs, ${M.w.phases} tokens a group. The newest token is <b>${tokName(M.run.ids[trace.query])}</b>, and attention sees the last ${M.w.window}. The model predicts output <b>${trace.n + 1}</b>.`],
  ['c-emb', () => `First the newest token becomes numbers: ${trace.queryRow.size} of the ${M.w.size} are 1. One of them, <b class="mono">phase[${trace.query % M.w.phases}]</b>, says where it sits in its group.`],
  ['c-att', () => { const i = M.heads.findIndex((h) => h.name === FETCH[mode][0]), hd = trace.heads[i], d = distance(trace.n % GROUP_OUT[mode], 0)
    return `Attention fetches what this output needs. Head <b>${FETCH[mode][0]}</b> looks <b>${d} back</b>, the same distance in every group; only that token's key matches (score ${hd.score}, the next best ${hd.rival.s}), so it gets weight ${hd.weight.toFixed(6)} and brings its bits. Click a head for its numbers.` }],
  ['c-att', () => { const d = distance(trace.n % GROUP_OUT[mode], 1)
    return `Head <b>${FETCH[mode][1]}</b> fetches the second symbol: <b>${d !== null ? d + ' back' : 'not needed for this output'}</b>. A missing or = symbol is how the model knows to pad or stop.` }],
  ['c-mlp', () => { const place = trace.n % GROUP_OUT[mode]
    return `This output is at place <b>${place}</b> of its group of ${GROUP_OUT[mode]}. The MLP's AND units copy each fetched bit to its place in the output: a solid wire carries a 1. Base64 never adds or carries, so wiring is all it takes.` }],
  ['c-out', () => `The output layer scores all ${M.w.tokens.length} tokens by how many bits agree. <b>${trace.name}</b> scores ${trace.scores[trace.ranked[0]]}; the next best, ${tokName(trace.ranked[1])}, ${trace.scores[trace.ranked[1]]}.`],
  ['c-next', () => `The winner is written and appended, and the next step reads it as the newest token. Your browser's own base64 says the same: ${wrong ? 'no!' : 'yes'}.`],
]
function tour(at) {
  tourAt = at
  if (at < 0 || at >= TOUR.length) { $('#tour').style.display = 'none'; $('#diagram').classList.remove('dim'); document.querySelectorAll('.col').forEach((c) => c.classList.remove('lit')); return }
  if (!trace) { step(); render() }
  const [col, text] = TOUR[at]
  $('#diagram').classList.add('dim')
  document.querySelectorAll('.col').forEach((c) => c.classList.toggle('lit', c.id === col))
  $('#tour-body').innerHTML = text()
  $('#tstep').textContent = `${at + 1} / ${TOUR.length}`
  const r = document.getElementById(col).getBoundingClientRect(), t = $('#tour')
  t.style.display = 'block'
  const left = Math.min(window.innerWidth - 376, Math.max(16, r.left + r.width / 2 - 180))
  t.style.left = left + 'px'
  const under = $('#diagram').getBoundingClientRect().bottom + 6      // below the diagram, so the lit column shows
  t.style.top = Math.max(16, Math.min(window.innerHeight - t.offsetHeight - 16, under)) + 'px'
}

// ---- wiring the controls ----------------------------------------------------------------------
function speed() { return +$('#speed').value }
async function main() {
  const w = await (await fetch('weights.json')).json()
  M = new Base64Transformer(w)
  $('#loading').hidden = true
  $('#wrap').hidden = $('#panels').hidden = $('#notes').hidden = false
  document.querySelectorAll('#mode button').forEach((b) => b.onclick = () => {
    if (b.dataset.m === mode) return
    const before = readInput()
    mode = b.dataset.m
    document.querySelectorAll('#mode button').forEach((x) => x.classList.toggle('on', x === b))
    // carry the text across: encoding "Man" -> decode its base64, decoding -> encode its bytes as text
    if (before.ok) $('#input').value = mode === 'decode' ? btoa(String.fromCharCode(...before.data)) : new TextDecoder().decode(Uint8Array.from([...atob(before.data)].map((c) => c.charCodeAt(0))))
    reset()
  })
  $('#input').oninput = reset
  $('#next').onclick = () => { stop(); step(); render(); if (tourAt >= 0) tour(tourAt) }
  $('#end').onclick = () => {
    stop()
    const go = () => { const t0 = performance.now(); while (performance.now() - t0 < 30) if (!step()) { render(); return } render(); setTimeout(go, 0) }
    go()
  }
  $('#play').onclick = () => {
    if (timer) return stop()
    $('#play').textContent = 'Pause'
    timer = setInterval(() => { const more = step(); render(); if (!more) stop() }, 1000 / speed())
  }
  $('#speed').oninput = () => { $('#speedv').textContent = speed() + '/s'; if (timer) { stop(); $('#play').click() } }
  $('#theme').onclick = () => {
    const dark = document.documentElement.dataset.theme !== 'dark' && !(document.documentElement.dataset.theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches)
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
    $('#theme').textContent = dark ? 'light' : 'dark'
    render()
  }
  if (matchMedia('(prefers-color-scheme: dark)').matches) $('#theme').textContent = 'light'   // offer the other one
  $('#tourbtn').onclick = () => tour(0)
  $('#tnext').onclick = () => tour(tourAt + 1 < TOUR.length ? tourAt + 1 : -1)
  $('#tprev').onclick = () => tour(Math.max(0, tourAt - 1))
  $('#tour .x').onclick = () => tour(-1)
  $('#card .x').onclick = () => $('#card').style.display = 'none'
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('#card').style.display = 'none'; tour(-1) } })
  document.querySelectorAll('.plus').forEach((b) => b.onclick = () => ({ seq: cardSeq, emb: cardSeq, att: () => cardHead(M.heads.findIndex((h) => h.name === FETCH[mode][0])), mlp: cardMlp, out: cardOut }[b.dataset.card])())
  $('#att').onclick = (e) => { const h = e.target.closest('.head'); if (h) cardHead(+h.dataset.head) }
  window.onresize = () => { drawRibbons(); if (tourAt >= 0) tour(tourAt) }
  reset()
}
main().catch((e) => { $('#loading').textContent = 'could not load weights.json: ' + e.message + ' (serve the folder: python -m http.server --directory web)' })
