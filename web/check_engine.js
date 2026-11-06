// The browser engine is the Python model: run it on the known answers (export.py) and on
// random inputs against Node's own base64. Run from the repo root:   node web/check_engine.js
const fs = require('fs')
const path = require('path')
const { Base64Transformer } = require('./engine.js')

const here = __dirname
const w = JSON.parse(fs.readFileSync(path.join(here, 'weights.json'), 'utf8'))
const known = JSON.parse(fs.readFileSync(path.join(here, 'known_answers.json'), 'utf8'))
const m = new Base64Transformer(w)

// 1  every score at every step equal to the Python model's
let worst = 0, steps = 0
for (const k of known) {
  m.start(k.mode, k.mode === 'encode' ? Uint8Array.from(k.input) : k.input)
  k.scores.forEach((row, i) => {
    const t = m.step()
    row.forEach((s, j) => { worst = Math.max(worst, Math.abs(s - t.scores[j])) })
    const want = k.tokens[i]
    if (t.token !== want) throw new Error(`${k.mode} ${JSON.stringify(k.input).slice(0, 30)}: step ${i} wrote ${t.name}`)
    steps++
  })
}
if (worst > 1e-6) throw new Error(`scores differ from Python by ${worst}`)
console.log(`1  ${known.length} known answers, ${steps} steps: the same tokens as Python, every score within ${worst.toExponential(1)}`)

// 2  whole answers against Node's base64, both directions
let rng = 7
const rand = (n) => { rng = (rng * 1103515245 + 12345) % 2147483648; return rng % n }
let answers = 0
for (let i = 0; i < 300; i++) {
  const data = Uint8Array.from({ length: rand(i < 100 ? 12 : 200) }, () => rand(256))
  const text = Buffer.from(data).toString('base64')
  const enc = m.answer('encode', data)
  const dec = m.answer('decode', text)
  if (enc !== text) throw new Error(`encode ${Buffer.from(data).toString('hex')}: ${enc} != ${text}`)
  if (Buffer.compare(Buffer.from(dec), Buffer.from(data)) !== 0) throw new Error(`decode ${text}`)
  answers += 2
}
console.log(`2  ${answers} whole answers (0-199 bytes) equal Node's base64, both directions`)

// 3  control: a planted mistake in the weights must be caught
const broken = JSON.parse(JSON.stringify(w))
const A = w.tokens.indexOf('A'), B = w.tokens.indexOf('B')
;[broken.unembed.c[A], broken.unembed.c[B]] = [broken.unembed.c[B], broken.unembed.c[A]]
for (const r in broken.unembed.U) broken.unembed.U[r] = broken.unembed.U[r].map(([c, v]) => [c === A ? B : c === B ? A : c, v])
const caught = new Base64Transformer(broken).answer('encode', Uint8Array.from([0, 0, 0])) !== 'AAAA'
if (!caught) throw new Error('the check passed an output layer with A and B swapped')
console.log("3  control: weights with 'A' and 'B' swapped are caught")
