import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend as RLegend } from 'recharts';
import { Play, Pause, RotateCcw, Brain, Database, Cpu, Edit3, Sparkles, BookOpen, Target, AlertTriangle, FastForward, Loader2, Zap, SkipForward, ChevronsRight, Activity, Award, Crosshair, GitCompare, Grid3x3, Network, Route, MessageSquare, Eye, Table2, GraduationCap, X, ChevronRight, ChevronLeft, Save, Trash2, FolderOpen, Download, Gauge } from 'lucide-react';

// ============ Math helpers ============
const randn = () => {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const topKIndices = (arr, k) => {
  const idx = [];
  for (let i = 0; i < arr.length; i++) idx.push(i);
  idx.sort((a, b) => Math.abs(arr[b]) - Math.abs(arr[a]));
  return idx.slice(0, Math.max(0, k));
};

// ============ Color palette ============
// Module-level palette switch. App sets this synchronously during render via
// setColorPalette() so all child renders in the same pass see the right value;
// there is no useEffect lag. The two options are 'default' (red/green diverging)
// and 'cbsafe' (Wong-style orange/blue diverging, deuteranopia & protanopia friendly).
let _palette = 'default';
const setColorPalette = (p) => { _palette = (p === 'cbsafe') ? 'cbsafe' : 'default'; };

const qColor = (q, scale) => {
  if (scale < 1e-6 || Math.abs(q) < 1e-6) return 'hsl(40, 15%, 96%)';
  const t = clamp(q / scale, -1, 1);
  if (_palette === 'cbsafe') {
    // Wong-style orange/blue diverging — both deutan/protan & tritan safe.
    if (t > 0) return `hsl(205, 75%, ${95 - t * 50}%)`;
    return `hsl(28, 85%, ${95 + t * 45}%)`;
  }
  if (t > 0) return `hsl(150, 60%, ${95 - t * 50}%)`;
  return `hsl(0, 65%, ${95 + t * 45}%)`;
};
const diffColor = (d, scale) => {
  if (scale < 1e-6) return 'hsl(220, 15%, 96%)';
  const t = clamp(d / scale, 0, 1);
  // Purple monochrome is already CB-safe; keep both palettes identical.
  return `hsl(280, 70%, ${95 - t * 55}%)`;
};
const errorColor = (e, scale) => {
  if (scale < 1e-6) return _palette === 'cbsafe' ? 'hsl(205, 60%, 96%)' : 'hsl(150, 60%, 95%)';
  const t = clamp(e / scale, 0, 1);
  if (_palette === 'cbsafe') {
    // Single-hue orange ramp (low error light, high error deep orange).
    return `hsl(${28 - t * 8}, ${70 + t * 20}%, ${95 - t * 55}%)`;
  }
  if (t < 0.5) return `hsl(${150 - t * 120}, 60%, ${95 - t * 30}%)`;
  return `hsl(${30 - (t - 0.5) * 60}, 70%, ${80 - (t - 0.5) * 30}%)`;
};
const visitColor = (n, maxN) => {
  if (maxN < 1 || n < 1) return 'hsl(40, 15%, 96%)';
  const t = clamp(Math.log(n + 1) / Math.log(maxN + 1), 0, 1);
  // Blue monochrome is CB-safe in both palettes.
  return `hsl(210, 75%, ${95 - t * 55}%)`;
};
const actColor = (v) => {
  const t = clamp(v / 3, 0, 1);
  const r = Math.round(30 + (96 - 30) * t);
  const g = Math.round(41 + (165 - 41) * t);
  const b = Math.round(59 + (250 - 59) * t);
  return `rgb(${r},${g},${b})`;
};
// Accent palette for the "matches optimal" (good) vs "disagrees" (bad) markers
// used on cell borders, the optimal badge, and the greedy-arrow text color.
// CB-safe uses Wong vermillion / bluish-green — orthogonal hues that all three
// major colorblindness types can distinguish.
const accentGood = () => _palette === 'cbsafe' ? '#009e73' : '#15803d';
const accentBad  = () => _palette === 'cbsafe' ? '#d55e00' : '#dc2626';

const PROBE_COLORS = ['#f97316', '#06b6d4', '#ec4899', '#84cc16'];

// ============ Neural Network ============
// Activation functions and their derivatives (derivative expressed in terms of
// the post-activation output, which is what backprop has on hand).
const ACTIVATIONS = {
  relu:   { fn: (z) => (z > 0 ? z : 0),               dFromOut: (a) => (a > 0 ? 1 : 0) },
  tanh:   { fn: (z) => Math.tanh(z),                  dFromOut: (a) => 1 - a * a },
  sigmoid:{ fn: (z) => 1 / (1 + Math.exp(-z)),        dFromOut: (a) => a * (1 - a) },
  leaky:  { fn: (z) => (z > 0 ? z : 0.01 * z),        dFromOut: (a) => (a > 0 ? 1 : 0.01) },
};
const ACT_NAMES = { relu: 'ReLU', tanh: 'tanh', sigmoid: 'sigmoid', leaky: 'Leaky ReLU' };

// Generalized fully-connected Q-network with an arbitrary list of hidden layers,
// each with its own activation. The output layer is linear (Q-values). The math
// is identical to the original two-layer version (verified by gradient check),
// just generalized to depth L. Weights for layer k are a flat (out_k × in_k)
// row-major Float32Array; Adam moments mirror them.
class DQN {
  // arch: { inSize, outSize, hidden: [{ size, act }] }  OR legacy (inSize, hSize, outSize)
  constructor(arch, legacyHSize, legacyOut) {
    if (typeof arch === 'number') {
      // legacy 2-hidden-layer signature, preserved for any old call sites
      arch = { inSize: arch, outSize: legacyOut, hidden: [{ size: legacyHSize, act: 'relu' }, { size: legacyHSize, act: 'relu' }] };
    }
    this.inSize = arch.inSize;
    this.outSize = arch.outSize;
    this.hidden = arch.hidden.map(h => ({ size: h.size, act: h.act || 'relu' }));
    // Layer dimensions: [in, h0, h1, ..., out]
    this.dims = [this.inSize, ...this.hidden.map(h => h.size), this.outSize];
    this.L = this.dims.length - 1; // number of weight layers
    // Backward-compat: first hidden size (used by some UI labels)
    this.hSize = this.hidden.length ? this.hidden[0].size : this.outSize;
    this.acts = [...this.hidden.map(h => h.act), 'linear']; // activation per weight layer; output linear

    this.W = []; this.b = []; this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    for (let k = 0; k < this.L; k++) {
      const nin = this.dims[k], nout = this.dims[k + 1];
      // He init for ReLU-family, Xavier for tanh/sigmoid.
      const act = this.acts[k];
      const scale = (act === 'tanh' || act === 'sigmoid') ? Math.sqrt(1 / nin) : Math.sqrt(2 / nin);
      const w = new Float32Array(nout * nin);
      for (let i = 0; i < w.length; i++) w[i] = randn() * scale;
      this.W.push(w); this.b.push(new Float32Array(nout));
      this.mW.push(new Float32Array(w.length)); this.vW.push(new Float32Array(w.length));
      this.mb.push(new Float32Array(nout)); this.vb.push(new Float32Array(nout));
    }
    this.t = 0;
  }
  forward(x) {
    // activations per layer; acts[0] = input, acts[k] = output of weight layer k-1
    const layerOuts = [x];
    let cur = x;
    for (let k = 0; k < this.L; k++) {
      const nin = this.dims[k], nout = this.dims[k + 1];
      const W = this.W[k], b = this.b[k];
      const out = new Float32Array(nout);
      const actName = this.acts[k];
      const act = actName === 'linear' ? null : ACTIVATIONS[actName].fn;
      for (let i = 0; i < nout; i++) {
        let s = b[i];
        const base = i * nin;
        for (let j = 0; j < nin; j++) s += W[base + j] * cur[j];
        out[i] = act ? act(s) : s;
      }
      layerOuts.push(out);
      cur = out;
    }
    const q = layerOuts[layerOuts.length - 1];
    // Back-compat aliases: a1/a2 are the first two hidden activations if present.
    return { x, layerOuts, q, a1: layerOuts[1] || null, a2: layerOuts[2] || null };
  }
  trainBatch(batch, lr) {
    const L = this.L;
    const dW = this.W.map(w => new Float32Array(w.length));
    const db = this.b.map(b => new Float32Array(b.length));
    let tl = 0;
    for (const sample of batch) {
      const x = sample.state, a = sample.action, t = sample.targetQ;
      const f = this.forward(x);
      const outs = f.layerOuts; // [x, h0, h1, ..., q]
      const err = f.q[a] - t;
      const ae = Math.abs(err);
      const huberGrad = ae <= 1 ? err : Math.sign(err);
      tl += ae <= 1 ? 0.5 * err * err : ae - 0.5;
      // delta at output layer: only the taken action carries gradient (linear out)
      let delta = new Float32Array(this.dims[L]);
      delta[a] = huberGrad;
      // backprop through layers L-1 .. 0
      for (let k = L - 1; k >= 0; k--) {
        const nin = this.dims[k], nout = this.dims[k + 1];
        const inAct = outs[k];          // input to this layer
        const Wk = this.W[k], dWk = dW[k], dbk = db[k];
        const prevDelta = new Float32Array(nin);
        for (let i = 0; i < nout; i++) {
          const d = delta[i];
          if (d === 0) continue;
          dbk[i] += d;
          const base = i * nin;
          for (let j = 0; j < nin; j++) {
            dWk[base + j] += d * inAct[j];
            prevDelta[j] += Wk[base + j] * d;
          }
        }
        // apply activation derivative of the PREVIOUS layer's output (layer k's input)
        if (k > 0) {
          const actName = this.acts[k - 1];
          const dFromOut = ACTIVATIONS[actName].dFromOut;
          for (let j = 0; j < nin; j++) prevDelta[j] *= dFromOut(inAct[j]);
        }
        delta = prevDelta;
      }
    }
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    const sc = 1 / batch.length;
    // Per-layer gradient L2 norm of the MEAN gradient (what Adam actually sees).
    // Read-only diagnostic — does not affect the update. Lets the UI surface
    // vanishing / exploding gradients, the key failure mode for deep nets.
    const gradNorms = [];
    let totalSq = 0;
    for (let k = 0; k < L; k++) {
      let s2 = 0;
      const dWk = dW[k], dbk = db[k];
      for (let i = 0; i < dWk.length; i++) { const g = dWk[i] * sc; s2 += g * g; }
      for (let i = 0; i < dbk.length; i++) { const g = dbk[i] * sc; s2 += g * g; }
      gradNorms.push(Math.sqrt(s2));
      totalSq += s2;
    }
    this.lastGradNorms = gradNorms;
    if (!this.gradNormHist) this.gradNormHist = [];
    this.gradNormHist.push(+Math.sqrt(totalSq).toFixed(5));
    if (this.gradNormHist.length > 120) this.gradNormHist.shift();
    const adam = (P, dP, m, v) => {
      for (let i = 0; i < P.length; i++) {
        const g = dP[i] * sc;
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        P[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
      }
    };
    for (let k = 0; k < L; k++) {
      adam(this.W[k], dW[k], this.mW[k], this.vW[k]);
      adam(this.b[k], db[k], this.mb[k], this.vb[k]);
    }
    return tl / batch.length;
  }
  copyFrom(o) {
    for (let k = 0; k < this.L; k++) { this.W[k].set(o.W[k]); this.b[k].set(o.b[k]); }
  }
  archSpec() {
    return { inSize: this.inSize, outSize: this.outSize, hidden: this.hidden.map(h => ({ size: h.size, act: h.act })) };
  }
  serialize() {
    return {
      arch: this.archSpec(), t: this.t,
      W: this.W.map(f32ToB64), b: this.b.map(f32ToB64),
    };
  }
  static deserialize(d) {
    // supports both new (arch + W/b arrays) and legacy (inSize/hSize/outSize + W1..W3) formats
    if (d.arch) {
      const net = new DQN(d.arch);
      net.t = d.t || 0;
      for (let k = 0; k < net.L; k++) { net.W[k].set(b64ToF32(d.W[k])); net.b[k].set(b64ToF32(d.b[k])); }
      return net;
    }
    const net = new DQN({ inSize: d.inSize, outSize: d.outSize, hidden: [{ size: d.hSize, act: 'relu' }, { size: d.hSize, act: 'relu' }] });
    net.t = d.t || 0;
    net.W[0].set(b64ToF32(d.W1)); net.b[0].set(b64ToF32(d.b1));
    net.W[1].set(b64ToF32(d.W2)); net.b[1].set(b64ToF32(d.b2));
    net.W[2].set(b64ToF32(d.W3)); net.b[2].set(b64ToF32(d.b3));
    return net;
  }
}

// Float32Array <-> base64 helpers for compact weight serialization.
function f32ToB64(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// ============ Environment ============
const W = 6, Hg = 6;
const ACTION_LABELS = ['↑', '→', '↓', '←'];
const ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
const DXY = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const MAX_STEPS = 60;
const STEP_REWARD = -0.04, GOAL_REWARD = 1.0, HAZARD_REWARD = -1.0;

// Module-level environment stochasticity. slipProb: probability the chosen action
// is overridden by a uniformly random one ("sticky"/noisy actions). rewardNoise:
// std-dev of Gaussian noise added to the per-step reward. Both default off, making
// the env deterministic (identical to before). Value iteration reads the same
// config so the "optimal" ground truth stays correct under stochasticity.
let _envCfg = { slipProb: 0, rewardNoise: 0 };
const setEnvConfig = (cfg) => { _envCfg = { slipProb: cfg.slipProb || 0, rewardNoise: cfg.rewardNoise || 0 }; };

const PRESETS = {
  open: { desc: 'Empty grid, easy sanity check.',
    grid: [[4,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,2]] },
  hazards: { desc: 'Walls and hazards force route planning.',
    grid: [[4,0,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,0,3],[0,1,1,0,0,0],[0,0,0,0,1,0],[0,0,3,0,0,2]] },
  maze: { desc: 'Long detours with multiple chokepoints.',
    grid: [[4,0,0,1,0,0],[1,1,0,1,0,1],[0,0,0,0,0,0],[0,1,1,1,1,0],[0,0,3,0,0,0],[1,0,1,0,1,2]] },
  cliff: { desc: 'Tempting shortcut along a row of hazards.',
    grid: [[4,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[3,3,3,3,3,0],[0,0,0,0,0,0],[0,0,0,0,0,2]] },
  detour: { desc: 'Goal next to start but blocked — forces a long detour.',
    grid: [[4,1,0,0,0,2],[0,1,0,1,1,1],[0,1,0,0,0,0],[0,1,0,1,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]] },
  trap: { desc: 'Direct path looks fine but ends in a trap.',
    grid: [[4,0,0,0,0,0],[1,1,1,1,1,0],[0,0,0,3,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0],[0,0,0,0,0,2]] },
};

const findStart = (g) => {
  for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) if (g[y][x] === 4) return [x, y];
  // No start marker (e.g. the user painted over it). Fall back to the first
  // walkable cell rather than [0,0] — if [0,0] happened to be a wall the agent
  // would spawn inside it, unable to move, and every episode would silently
  // time out forever.
  for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) if (g[y][x] === 0) return [x, y];
  return [0, 0];
};

class Env {
  constructor(g) {
    this.grid = g.map(r => [...r]);
    [this.sx, this.sy] = findStart(g);
    this.reset();
  }
  reset() { this.x = this.sx; this.y = this.sy; this.t = 0; return this.encode(); }
  encode() { const s = new Float32Array(W * Hg); s[this.y * W + this.x] = 1; return s; }
  encodeAt(x, y) { const s = new Float32Array(W * Hg); s[y * W + x] = 1; return s; }
  step(a) {
    // Slip: with probability slipProb, replace the intended action with a random
    // one. This makes transitions stochastic — the regime where vanilla DQN's
    // overestimation bias actually hurts.
    let act = a;
    if (_envCfg.slipProb > 0 && Math.random() < _envCfg.slipProb) act = Math.floor(Math.random() * 4);
    const [dx, dy] = DXY[act];
    const nx = this.x + dx, ny = this.y + dy;
    let r = STEP_REWARD, terminal = false, truncated = false;
    if (nx >= 0 && nx < W && ny >= 0 && ny < Hg && this.grid[ny][nx] !== 1) { this.x = nx; this.y = ny; }
    const c = this.grid[this.y][this.x];
    if (c === 2) { r = GOAL_REWARD; terminal = true; }
    else if (c === 3) { r = HAZARD_REWARD; terminal = true; }
    // Reward noise: Gaussian jitter on the step reward (terminal rewards included).
    if (_envCfg.rewardNoise > 0) r += randn() * _envCfg.rewardNoise;
    this.t++;
    if (!terminal && this.t >= MAX_STEPS) truncated = true;
    return { s: this.encode(), r, terminal, truncated };
  }
}

// ============ Value Iteration ============
function valueIteration(grid, gamma, slipProb = 0) {
  const V = new Float32Array(W * Hg);
  const idx = (x, y) => y * W + x;
  // Single-action outcome: where you land, the reward, and whether it's terminal.
  // Reward noise has mean 0 so it does not affect expected value; only slip does.
  const outcome = (x, y, a) => {
    const [dx, dy] = DXY[a];
    let nx = x + dx, ny = y + dy;
    if (nx < 0 || nx >= W || ny < 0 || ny >= Hg || grid[ny][nx] === 1) { nx = x; ny = y; }
    const nc = grid[ny][nx];
    let r = STEP_REWARD, term = false;
    if (nc === 2) { r = GOAL_REWARD; term = true; }
    else if (nc === 3) { r = HAZARD_REWARD; term = true; }
    return { nx, ny, r, term };
  };
  // Expected Q of intending action `a` under slip: with prob (1-p) the intended
  // action executes; with prob p a uniformly random action executes. So the
  // effective distribution over executed actions is:
  //   intended: (1-p) + p/4 ;  each other: p/4.
  const qExpected = (x, y, a) => {
    if (slipProb <= 0) {
      const { nx, ny, r, term } = outcome(x, y, a);
      return r + gamma * (term ? 0 : V[idx(nx, ny)]);
    }
    let q = 0;
    for (let b = 0; b < 4; b++) {
      const prob = (b === a ? (1 - slipProb) : 0) + slipProb / 4;
      const { nx, ny, r, term } = outcome(x, y, b);
      q += prob * (r + gamma * (term ? 0 : V[idx(nx, ny)]));
    }
    return q;
  };
  for (let iter = 0; iter < 1000; iter++) {
    let maxDelta = 0;
    const newV = new Float32Array(W * Hg);
    for (let y = 0; y < Hg; y++) {
      for (let x = 0; x < W; x++) {
        const c = grid[y][x];
        if (c === 1 || c === 2 || c === 3) { newV[idx(x, y)] = 0; continue; }
        let bestQ = -Infinity;
        for (let a = 0; a < 4; a++) { const q = qExpected(x, y, a); if (q > bestQ) bestQ = q; }
        newV[idx(x, y)] = bestQ;
        const d = Math.abs(bestQ - V[idx(x, y)]);
        if (d > maxDelta) maxDelta = d;
      }
    }
    V.set(newV);
    if (maxDelta < 1e-7) break;
  }
  const policy = new Int8Array(W * Hg).fill(-1);
  for (let y = 0; y < Hg; y++) {
    for (let x = 0; x < W; x++) {
      const c = grid[y][x];
      if (c !== 0 && c !== 4) continue;
      let bestA = 0, bestQ = -Infinity;
      for (let a = 0; a < 4; a++) { const q = qExpected(x, y, a); if (q > bestQ) { bestQ = q; bestA = a; } }
      policy[idx(x, y)] = bestA;
    }
  }
  return { V, policy };
}

// ============ Agents ============
// Turn a hyperparameter object into a network architecture spec. If hp.layers is
// present (the custom-architecture feature) it's used directly; otherwise we fall
// back to the classic two-hidden-layer ReLU net sized by hp.hSize.
const archFromHp = (hp) => {
  const hidden = (hp.layers && hp.layers.length)
    ? hp.layers.map(l => ({ size: l.size, act: l.act || 'relu' }))
    : [{ size: hp.hSize, act: 'relu' }, { size: hp.hSize, act: 'relu' }];
  return { inSize: W * Hg, outSize: 4, hidden };
};
// True if the architecture implied by hp differs from a live network's structure
// (used to prompt "reset to rebuild"). Compares layer count, widths, activations.
const archDiffers = (hp, net) => {
  const want = archFromHp(hp).hidden;
  const have = net.hidden || [];
  if (want.length !== have.length) return true;
  for (let i = 0; i < want.length; i++) {
    if (want[i].size !== have[i].size || want[i].act !== have[i].act) return true;
  }
  return false;
};

// Read the agent's greedy max-Q for every cell into a flat W*Hg array. Used for
// the value-propagation (#1) delta snapshots. Cheap: 36 forward passes.
// IMPORTANT: must be read-only. For the tabular agent, q.forward() would create
// a zero Q-row for any unvisited cell (polluting the "visited states" semantics
// of the Q-table), so we read existing rows directly without creating them.
const readMaxQGrid = (agent) => {
  const out = new Float32Array(W * Hg);
  const isTab = agent instanceof TabularAgent;
  for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
    if (isTab) {
      const row = agent.qTable.get(`${x},${y}`);
      out[y * W + x] = row ? Math.max(row[0], row[1], row[2], row[3]) : 0;
    } else {
      const s = new Float32Array(W * Hg); s[y * W + x] = 1;
      const f = agent.q.forward(s);
      out[y * W + x] = Math.max(f.q[0], f.q[1], f.q[2], f.q[3]);
    }
  }
  return out;
};

class Agent {
  constructor(hp) {
    this.hp = { ...hp };
    const arch = archFromHp(hp);
    this.q = new DQN(arch);
    this.tgt = new DQN(arch);
    this.tgt.copyFrom(this.q);
    this.buffer = [];
    this.eps = hp.epsStart;
    this.trainSteps = 0;
  }
  act(s) {
    const f = this.q.forward(s);
    if (Math.random() < this.eps) return { a: Math.floor(Math.random() * 4), explore: true, q: Array.from(f.q), a1: f.a1, a2: f.a2 };
    let best = 0;
    for (let i = 1; i < 4; i++) if (f.q[i] > f.q[best]) best = i;
    return { a: best, explore: false, q: Array.from(f.q), a1: f.a1, a2: f.a2 };
  }
  actGreedy(s) {
    const f = this.q.forward(s);
    let best = 0;
    for (let i = 1; i < 4; i++) if (f.q[i] > f.q[best]) best = i;
    return best;
  }
  remember(s, a, r, s2, terminal) {
    this.buffer.push({ s, a, r, s2, terminal });
    if (this.buffer.length > this.hp.bufferSize) this.buffer.shift();
  }
  learn() {
    if (this.buffer.length < this.hp.batchSize) return null;
    const batch = [];
    for (let i = 0; i < this.hp.batchSize; i++) batch.push(this.buffer[Math.floor(Math.random() * this.buffer.length)]);
    let totalTd = 0, displaySample = null;
    const tb = batch.map(({ s, a, r, s2, terminal }, idx) => {
      let target = r, maxNextQ = 0, maxNextA = -1;
      if (!terminal) {
        if (this.hp.doubleDQN) {
          // Double DQN: SELECT the next action with the online network, then
          // EVALUATE that action with the target network. Decoupling selection
          // from evaluation removes the systematic overestimation (maximization
          // bias) that vanilla DQN's single max introduces.
          const fOnline = this.q.forward(s2);
          let sel = 0;
          for (let i = 1; i < 4; i++) if (fOnline.q[i] > fOnline.q[sel]) sel = i;
          const fTarget = this.tgt.forward(s2);
          maxNextA = sel;
          maxNextQ = fTarget.q[sel];
        } else {
          // Vanilla DQN: the target network both selects and evaluates (a single
          // max over the target net's Q-values).
          const f = this.tgt.forward(s2);
          let m = f.q[0]; maxNextA = 0;
          for (let i = 1; i < 4; i++) if (f.q[i] > m) { m = f.q[i]; maxNextA = i; }
          maxNextQ = m;
        }
        target += this.hp.gamma * maxNextQ;
      }
      const cur = this.q.forward(s);
      totalTd += Math.abs(cur.q[a] - target);
      if (idx === 0) displaySample = { state: s, action: a, reward: r, nextState: s2, terminal, oldQ: cur.q[a], maxNextQ, maxNextA, target, tdError: target - cur.q[a], doubleDQN: !!this.hp.doubleDQN };
      return { state: s, action: a, targetQ: target };
    });
    const loss = this.q.trainBatch(tb, this.hp.lr);
    this.trainSteps++;
    let sync = false;
    if (this.trainSteps % this.hp.targetUpdate === 0) { this.tgt.copyFrom(this.q); sync = true; }
    if (displaySample) displaySample.newQ = this.q.forward(displaySample.state).q[displaySample.action];
    return { loss, sync, avgTd: totalTd / batch.length, bellman: displaySample };
  }
  decayEps() { this.eps = Math.max(this.hp.epsMin, this.eps * this.hp.epsDecay); }
  serialize() {
    // Replay buffer: store one-hot states as their active index to stay compact.
    const idxOf = (s) => { for (let i = 0; i < s.length; i++) if (s[i] > 0.5) return i; return -1; };
    const buffer = this.buffer.map(({ s, a, r, s2, terminal }) => [idxOf(s), a, r, idxOf(s2), terminal ? 1 : 0]);
    return { kind: 'dqn', hp: this.hp, eps: this.eps, trainSteps: this.trainSteps, q: this.q.serialize(), buffer };
  }
  static deserialize(d) {
    const ag = new Agent(d.hp);
    ag.q = DQN.deserialize(d.q);
    ag.tgt = DQN.deserialize(d.q); // re-sync target to online; cheap and avoids storing both
    ag.eps = d.eps;
    ag.trainSteps = d.trainSteps || 0;
    const N = W * Hg;
    const oneHot = (i) => { const s = new Float32Array(N); if (i >= 0) s[i] = 1; return s; };
    ag.buffer = (d.buffer || []).map(([si, a, r, s2i, term]) => ({ s: oneHot(si), a, r, s2: oneHot(s2i), terminal: !!term }));
    return ag;
  }
}

class TabularAgent {
  constructor(hp) {
    this.hp = { ...hp };
    this.qTable = new Map();
    this.eps = hp.epsStart;
    this.trainSteps = 0;
    this.lastTransition = null;
    this.q = { hSize: 0, forward: (s) => { const [x, y] = TabularAgent.posOf(s); return { q: this.getRow(x, y), a1: null, a2: null }; } };
    this.tgt = this.q;
    this.buffer = [];
  }
  static posOf(s) { for (let i = 0; i < s.length; i++) if (s[i] > 0.5) return [i % W, Math.floor(i / W)]; return [-1, -1]; }
  getRow(x, y) {
    const key = `${x},${y}`;
    if (!this.qTable.has(key)) this.qTable.set(key, new Float32Array(4));
    return this.qTable.get(key);
  }
  act(s) {
    const f = this.q.forward(s);
    if (Math.random() < this.eps) return { a: Math.floor(Math.random() * 4), explore: true, q: Array.from(f.q), a1: null, a2: null };
    let best = 0;
    for (let i = 1; i < 4; i++) if (f.q[i] > f.q[best]) best = i;
    return { a: best, explore: false, q: Array.from(f.q), a1: null, a2: null };
  }
  actGreedy(s) {
    const f = this.q.forward(s);
    let best = 0;
    for (let i = 1; i < 4; i++) if (f.q[i] > f.q[best]) best = i;
    return best;
  }
  remember(s, a, r, s2, terminal) { this.lastTransition = { s, a, r, s2, terminal }; }
  learn() {
    if (!this.lastTransition) return null;
    const { s, a, r, s2, terminal } = this.lastTransition;
    const [sx, sy] = TabularAgent.posOf(s);
    const qs = this.getRow(sx, sy);
    const oldQ = qs[a];
    let maxNextQ = 0, maxNextA = -1;
    if (!terminal) {
      const [s2x, s2y] = TabularAgent.posOf(s2);
      const qs2 = this.getRow(s2x, s2y);
      maxNextQ = qs2[0]; maxNextA = 0;
      for (let i = 1; i < 4; i++) if (qs2[i] > maxNextQ) { maxNextQ = qs2[i]; maxNextA = i; }
    }
    const target = r + (terminal ? 0 : this.hp.gamma * maxNextQ);
    const tdError = target - oldQ;
    const newQ = oldQ + this.hp.alpha * tdError;
    qs[a] = newQ;
    this.trainSteps++;
    return { loss: 0.5 * tdError * tdError, sync: false, avgTd: Math.abs(tdError),
      bellman: { state: s, action: a, reward: r, nextState: s2, terminal, oldQ, maxNextQ, maxNextA, target, tdError, newQ } };
  }
  decayEps() { this.eps = Math.max(this.hp.epsMin, this.eps * this.hp.epsDecay); }
  serialize() {
    const table = [];
    for (const [key, row] of this.qTable.entries()) table.push([key, row[0], row[1], row[2], row[3]]);
    return { kind: 'tabular', hp: this.hp, eps: this.eps, trainSteps: this.trainSteps, table };
  }
  static deserialize(d) {
    const ag = new TabularAgent(d.hp);
    ag.eps = d.eps;
    ag.trainSteps = d.trainSteps || 0;
    for (const [key, q0, q1, q2, q3] of (d.table || [])) {
      ag.qTable.set(key, Float32Array.from([q0, q1, q2, q3]));
    }
    return ag;
  }
}
const deserializeAgent = (d) => d.kind === 'tabular' ? TabularAgent.deserialize(d) : Agent.deserialize(d);
const DEFAULT_HP = { epsStart: 1.0, epsMin: 0.05, epsDecay: 0.99, gamma: 0.95, alpha: 0.5, lr: 0.005, batchSize: 32, bufferSize: 2000, targetUpdate: 100, hSize: 32, doubleDQN: false, layers: [{ size: 32, act: 'relu' }, { size: 32, act: 'relu' }] };
const DEFAULT_HP_B = { ...DEFAULT_HP, lr: 0.001, gamma: 0.85, alpha: 0.1, doubleDQN: true, layers: [{ size: 32, act: 'relu' }, { size: 32, act: 'relu' }] };
const makeAgent = (algorithm, hp) => algorithm === 'tabular' ? new TabularAgent(hp) : new Agent(hp);

function buildNarration(action, transition, learn, algorithm, eps, bufferSize) {
  const aLabel = ACTION_NAMES[action.a];
  const exploreText = action.explore ? '⚄ explored randomly' : '★ greedy choice';
  const moved = transition.sx !== transition.dx || transition.sy !== transition.dy;
  const motionText = moved ? `moved ${aLabel} → (${transition.dx},${transition.dy})` : `tried ${aLabel}, blocked, stayed put`;
  let text = `At (${transition.sx},${transition.sy}), ε=${eps.toFixed(2)}: ${exploreText}, ${motionText}. Reward ${transition.reward >= 0 ? '+' : ''}${transition.reward.toFixed(2)}.`;
  if (algorithm === 'tabular' && learn?.bellman) {
    const b = learn.bellman;
    text += ` TD δ=${b.tdError.toFixed(3)}. Q: ${b.oldQ.toFixed(3)} → ${b.newQ.toFixed(3)}.`;
  } else if (algorithm === 'dqn') {
    if (learn) text += ` Buffer: ${bufferSize}. Adam step, loss=${learn.loss.toFixed(3)}.`;
    else text += ` Buffer: ${bufferSize} (filling).`;
  }
  if (transition.terminal) text += transition.reward > 0 ? ' 🚩 Goal!' : ' 💀 Hazard!';
  return text;
}

// ============ Tutorial ============
const TUTORIAL_STEPS = [
  { title: 'Welcome to the Q-Learning Visualizer',
    body: 'A reinforcement learning agent will learn to navigate a grid world from scratch. This quick tour shows you each part. Use Next/Back to move through, or Skip to dismiss.' },
  { target: '[data-tutor="grid"]', position: 'right', title: 'The grid world',
    body: 'The blue dot is the agent. 🚩 is the goal (+1 reward). 💀 is a hazard (−1). Each step costs −0.04 to encourage finding short paths. Walls block movement.' },
  { target: '[data-tutor="grid"]', position: 'right', title: 'Q-values per cell',
    body: 'Each cell has four wedges — the agent\u2019s estimate of Q(s,a) for ↑→↓←. Green = high value, red = low. The center letter is the action it would pick now. They start gray because the agent knows nothing yet.' },
  { target: '[data-tutor="step-btn"]', position: 'top', title: 'Take one step',
    body: 'Click Step to advance one transition: agent picks an action, takes it, observes a reward, and updates its Q-values. Click it a few times now and watch the cells change.' },
  { target: '[data-tutor="bellman"]', tab: 'status', position: 'left', title: 'The Bellman update',
    body: 'After every step, this panel shows the math. Target = reward + γ·max(future Q). TD error δ = target − Q(s,a). The network (or table) nudges Q toward the target proportional to δ.' },
  { target: '[data-tutor="fast-train"]', position: 'top', title: 'Train faster',
    body: 'Step-by-step is slow. Animate runs continuously with visualization. +5k steps trains silently in the background — try clicking it now to see the policy actually emerge.' },
  { target: '[data-tutor="stats"]', tab: 'status', position: 'left', title: 'Are we converging?',
    body: 'Policy match shows the fraction of cells where the agent\u2019s greedy action matches the optimal action (computed by value iteration). 100% means the agent has fully converged. V*(start) is the optimal return from the start state.' },
  { target: '[data-tutor="view-modes"]', position: 'bottom', title: 'Different views of the grid',
    body: 'Q = the agent\u2019s learned values. V* = ground truth from value iteration. Err = where the agent is still wrong. Visits = exploration heatmap. Δtgt (DQN only) = lag between online and target networks.' },
  { target: '[data-tutor="rollout-btn"]', position: 'top', title: 'Test the policy',
    body: 'Click Rollout to watch the agent execute its current greedy policy from start to goal — no exploration, no learning. The clearest way to see whether learning worked.' },
  { target: '[data-tutor="algo-toggle"]', position: 'bottom', title: 'Tabular vs Deep Q',
    body: 'DQN uses a small neural network to predict Q-values. Tabular Q uses a literal lookup table. Both solve this task, but with very different mechanics. Switch any time to compare.' },
  { target: '[data-tutor="presets"]', position: 'top', title: 'Curriculum presets',
    body: 'Each preset illustrates a different concept: cliff (risk), detour (value propagation), trap (bootstrapping). Or click Edit to design your own grid.' },
  { title: "You're all set",
    body: 'Explore further with Probes (track cells over time), Compare A/B (two agents side-by-side), Narrate (plain-English step commentary), and Technical mode (network internals, replay buffer, Q-table). Want the theory? Open “The math” walkthrough up top for a guided derivation of Q-learning. Have fun.' },
];

// ============ Guided auto-demo ============
// A hands-free walkthrough that DRIVES the app: it resets, trains, switches views
// and inspector tabs, and runs a rollout, narrating each part and spotlighting it.
// Unlike the manual tours, it auto-advances. `action` tells the App driver what to
// do; `dwell` is how long to linger on read-only steps (training steps advance when
// training finishes). `target`/`tab`/`viewMode` set what's shown.
const GUIDED_DEMO = [
  { title: 'Guided demo', body: 'Sit back — I\u2019ll run a full example for you: start from a blank agent, train it, explore how it learns, and watch it solve the grid. You can pause or stop anytime.',
    action: { type: 'reset' }, dwell: 4200 },
  { target: '[data-tutor="grid"]', position: 'right', title: 'A blank slate', viewMode: 'q',
    body: 'Here\u2019s the world. Blue dot = agent, 🚩 = goal (+1), 💀 = hazard (−1). Every cell\u2019s four wedges are the agent\u2019s Q-value estimates — all gray now, because it knows nothing yet.',
    dwell: 6000 },
  { target: '[data-tutor="step-btn"]', position: 'top', title: 'One step at a time',
    body: 'A single step: the agent picks an action, moves, gets a reward, and nudges its Q-values. Watch a few cells light up as I take five steps.',
    action: { type: 'steps', n: 5 }, dwell: 5200 },
  { target: '[data-tutor="bellman"]', tab: 'status', position: 'left', title: 'The update rule',
    body: 'Each step runs this: target = r + γ·max(next Q), and the value moves toward it by the TD error. This is the entire learning signal — no labels, just bootstrapping.',
    dwell: 6500 },
  { target: '[data-tutor="fast-train"]', position: 'top', title: 'Now train for real',
    body: 'Stepping by hand is slow. I\u2019ll train 4,000 steps in the background — watch the grid fill with color as value propagates outward from the goal.',
    action: { type: 'train', steps: 4000 } },
  { target: '[data-tutor="grid"]', position: 'right', title: 'Value flowing back', viewMode: 'vprop',
    body: 'The Flow view colors each cell by how much its value just changed. Blue = value arriving. Early in training you can literally see credit spreading backward from the goal — the core of how RL assigns reward.',
    dwell: 7000 },
  { target: '[data-tutor="grid"]', position: 'right', title: 'Where it has been', viewMode: 'trajectories',
    body: 'The Paths view overlays recent episodes — green reached the goal, red hit a hazard, amber timed out. As the policy sharpens, the routes converge onto the efficient path.',
    dwell: 7000 },
  { target: '[data-tutor="grid"]', position: 'right', title: 'How wrong is it still?', viewMode: 'verror',
    body: 'The Err view compares the agent\u2019s values to the true optimum from value iteration. Green = accurate, red = still off. Most of the grid should be green by now.',
    dwell: 6500 },
  { tab: 'charts', title: 'The learning curves', target: '[data-tutor="match-chart"]', position: 'left',
    body: 'Policy match is the fraction of cells where the agent\u2019s greedy action equals the optimal one. Climbing toward 100% is the clearest sign learning is working.',
    dwell: 6800 },
  { target: '[data-tutor="grid"]', position: 'right', title: 'Polishing it off', viewMode: 'q',
    body: 'A short top-up of training to finish converging, back on the Q-value view so you can see the arrows settle into a consistent policy.',
    action: { type: 'train', steps: 3000 } },
  { target: '[data-tutor="rollout-btn"]', position: 'top', title: 'Put it to the test',
    body: 'Finally, a rollout: the agent runs its learned greedy policy from start to goal — no exploration, no learning. The proof that it solved the task.',
    action: { type: 'rollout' }, dwell: 5500 },
  { title: 'That\u2019s the whole loop', body: 'From a blank agent to a solved grid: act → observe → update, repeated thousands of times, with value flowing back from the goal. Everything I just did is available to you by hand — explore the tabs, try Compare mode, or design your own grid. Enjoy!',
    dwell: 9000 },
];

// Technical-mode tour for Deep Q-Networks. Targets only render in technical mode
// with the DQN algorithm selected, so the launcher is gated accordingly.
const DQN_TECH_TOUR = [
  { title: 'Inside the Deep Q-Network',
    body: 'You\u2019re in Technical mode with the DQN agent. This short tour walks through every internal panel — the network, the replay buffer, the loss curve, and how they fit together. Each step switches the inspector beside the grid to the right view. Use ← → or Next/Back.' },
  { target: '[data-tutor="bellman"]', tab: 'status', position: 'left', title: '1 · The TD target',
    body: 'Every learning step starts here. The target is r + γ·maxₐ′ Q(s′,a′) computed from the target network. The TD error δ = target − Q(s,a) is what the network is trained to shrink. This is the supervision signal — there are no labels, the agent bootstraps from its own next-state estimate.' },
  { target: '[data-tutor="network"]', tab: 'model', position: 'top', title: '2 · The live network Q(s,a;θ)',
    body: 'A multilayer perceptron maps the one-hot state (36 inputs) through its hidden layers to four Q-values. Brighter neurons = stronger activation for the current state; line thickness shows the strongest weighted paths. You can redesign these hidden layers yourself in the Settings tab — add layers, resize them, change activations. This replaces the lookup table: it generalizes across nearby states instead of storing each one.' },
  { target: '[data-tutor="replay"]', tab: 'memory', position: 'right', title: '3 · Experience replay',
    body: 'Transitions (s, a, r, s′, done) are stored in a buffer. Each training step samples a random minibatch from it rather than learning from the latest step alone. This breaks the correlation between consecutive samples and lets each experience be reused many times — both critical for stable deep RL.' },
  { target: '[data-tutor="hyperparams"]', tab: 'settings', position: 'left', title: '4 · The knobs that matter',
    body: 'Learning rate scales each gradient step. Batch size and replay capacity govern how much data each update sees. Target sync sets how often θ⁻ copies θ — too frequent and targets chase themselves, too rare and they go stale. The architecture editor lets you redesign the network itself.' },
  { target: '[data-tutor="loss-chart"]', tab: 'charts', position: 'top', title: '5 · The loss curve',
    body: 'Mean squared TD error per update. Unlike supervised learning it rarely goes monotonically to zero — the target moves as the network learns, and exploration keeps feeding surprises. A downward trend with bounded spikes around target syncs is healthy.' },
  { target: '[data-tutor="eps-chart"]', tab: 'charts', position: 'top', title: '6 · Exploration schedule',
    body: 'ε decays from its start value toward the floor. Early on the agent acts mostly randomly to fill the buffer with diverse experience; later it increasingly exploits its learned Q-values. The loss and policy-match curves only stabilize once ε is low.' },
  { target: '[data-tutor="match-chart"]', tab: 'charts', position: 'top', title: '7 · Are we converging?',
    body: 'Policy match is the fraction of cells where the network\u2019s greedy action equals the optimal action from value iteration. As Q(·;θ) approaches Q*, this climbs toward 100%. It\u2019s the clearest single signal that deep Q-learning is actually working.' },
  { title: 'That\u2019s the full loop',
    body: 'Act (ε-greedy) → store in replay → sample a batch → compute TD targets from θ⁻ → gradient step on θ → periodically sync θ⁻. Switch inspector tabs anytime to watch a different part. Switch to Tabular Q for a tour of the table-based version.' },
];

// Technical-mode tour for tabular Q-learning.
const TABULAR_TECH_TOUR = [
  { title: 'Inside tabular Q-learning',
    body: 'You\u2019re in Technical mode with the tabular agent. This tour walks through the mechanics: the literal Q-table, the per-step Bellman update, and the curves that show convergence. Each step switches the inspector beside the grid. Use ← → or Next/Back.' },
  { target: '[data-tutor="bellman"]', tab: 'status', position: 'left', title: '1 · The Q-update, step by step',
    body: 'After each transition the agent updates one cell of one row: Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′ Q(s′,a′) − Q(s,a)]. This panel shows the target, the old value, the TD error δ, and the new value. No network, no gradients — just a direct nudge of a single number.' },
  { target: '[data-tutor="qtable"]', tab: 'model', position: 'top', title: '2 · The Q-table itself',
    body: 'One row per visited state, four columns for ↑ → ↓ ←. The bold number is the greedy action\u2019s value; the arrow turns red when it disagrees with the optimal policy. This is the entire "model" — a literal lookup table with no generalization, so every state must be visited to be learned.' },
  { target: '[data-tutor="hyperparams"]', tab: 'settings', position: 'left', title: '3 · The knobs that matter',
    body: 'α (learning rate) sets how far each update moves Q toward the target — high α learns fast but noisily, low α is slow but stable. γ is the discount horizon. ε decay controls how quickly exploration gives way to exploitation. There\u2019s no network here, so no batch size or target network.' },
  { target: '[data-tutor="eps-chart"]', tab: 'charts', position: 'top', title: '4 · Exploration schedule',
    body: 'ε decays from start toward the floor. Tabular Q-learning needs every state-action pair visited often enough, so early exploration matters: with too little, some table cells never get updated and the policy stays wrong there.' },
  { target: '[data-tutor="match-chart"]', tab: 'charts', position: 'top', title: '5 · Are we converging?',
    body: 'Policy match is the fraction of cells where the table\u2019s greedy action equals the optimal action from value iteration. With enough exploration and a sensible α, tabular Q-learning is provably guaranteed to reach 100%.' },
  { title: 'That\u2019s the tabular loop',
    body: 'Act (ε-greedy) → observe (r, s′) → update one table cell toward the TD target → decay ε. Simple, exact, and convergent — but it doesn\u2019t scale to large state spaces, which is exactly why DQN swaps the table for a network. Switch to Deep Q for that tour.' },
];

function Tutorial({ step, setStep, onClose, steps = TUTORIAL_STEPS, accent = 'blue', onStepEnter }) {
  const [box, setBox] = useState(null);
  const stepData = steps[step];

  // Notify the host when a step is entered (e.g. to switch the inspector tab so
  // the step's target element is actually mounted). Fires before the spotlight
  // tracker, which polls via rAF and will pick up the element once it mounts.
  useEffect(() => {
    if (onStepEnter && stepData) onStepEnter(stepData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  useEffect(() => {
    if (!stepData?.target) { setBox(null); return; }
    let raf = 0, cancelled = false, last = null;
    const el0 = document.querySelector(stepData.target);
    if (el0) {
      const reduce = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el0.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center', inline: 'nearest' });
    }
    // Track the target every frame; only commit state when it actually moved.
    // 60fps follow makes the spotlight glide along with the smooth-scroll
    // instead of the steppy ~12fps interval it used before.
    const tick = () => {
      if (cancelled) return;
      const el = document.querySelector(stepData.target);
      if (el) {
        const r = el.getBoundingClientRect();
        if (!last ||
            Math.abs(r.top - last.top) > 0.5 || Math.abs(r.left - last.left) > 0.5 ||
            Math.abs(r.width - last.width) > 0.5 || Math.abs(r.height - last.height) > 0.5) {
          last = { top: r.top, left: r.left, width: r.width, height: r.height };
          setBox(last);
        }
      } else if (last) { last = null; setBox(null); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelled = true; cancelAnimationFrame(raf); };
  }, [stepData?.target]);

  const isLast = step === steps.length - 1;
  const isFirst = step === 0;
  const centered = !stepData?.target;

  // Keyboard navigation: ←/→ to move, Esc to close.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); isLast ? onClose() : setStep(step + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); if (!isFirst) setStep(step - 1); }
      else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, isLast, isFirst, onClose, setStep]);

  // Compute tooltip position
  const tipWidth = 340;
  let tipStyle = { width: tipWidth, transition: 'top 0.28s cubic-bezier(0.4,0,0.2,1), left 0.28s cubic-bezier(0.4,0,0.2,1), right 0.28s cubic-bezier(0.4,0,0.2,1), bottom 0.28s cubic-bezier(0.4,0,0.2,1)' };
  if (centered) {
    tipStyle = { ...tipStyle, top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
  } else if (box) {
    const vw = typeof window !== 'undefined' ? window.innerWidth : 1200;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 800;
    const cx = box.left + box.width / 2;
    const cy = box.top + box.height / 2;
    let pos = stepData.position || 'bottom';
    // Auto-fallback if no room on the chosen side
    if (pos === 'right' && box.left + box.width + tipWidth + 32 > vw) pos = 'left';
    if (pos === 'left' && box.left - tipWidth - 32 < 0) pos = 'bottom';
    if (pos === 'top' && box.top < 220) pos = 'bottom';
    if (pos === 'bottom' && box.top + box.height + 220 > vh) pos = 'top';

    if (pos === 'top') tipStyle = { ...tipStyle, bottom: vh - box.top + 16, left: clamp(cx - tipWidth / 2, 12, vw - tipWidth - 12) };
    else if (pos === 'bottom') tipStyle = { ...tipStyle, top: box.top + box.height + 16, left: clamp(cx - tipWidth / 2, 12, vw - tipWidth - 12) };
    else if (pos === 'left') tipStyle = { ...tipStyle, right: vw - box.left + 16, top: clamp(cy - 100, 12, vh - 220) };
    else if (pos === 'right') tipStyle = { ...tipStyle, left: box.left + box.width + 16, top: clamp(cy - 100, 12, vh - 220) };
  } else {
    tipStyle = { ...tipStyle, top: 80, left: '50%', transform: 'translateX(-50%)' };
  }

  // Accent palette so the main tour, the tabular tour, and the DQN tour read
  // as distinct flavors while sharing one component.
  const ACCENTS = {
    blue:    { rgb: '96,165,250',  card: 'border-blue-500',    title: 'text-blue-300',    btn: 'bg-blue-600 hover:bg-blue-500',       dotOn: 'bg-blue-500',    dotPast: 'bg-blue-700' },
    emerald: { rgb: '52,211,153',  card: 'border-emerald-500', title: 'text-emerald-300', btn: 'bg-emerald-600 hover:bg-emerald-500', dotOn: 'bg-emerald-500', dotPast: 'bg-emerald-700' },
    sky:     { rgb: '56,189,248',  card: 'border-sky-500',     title: 'text-sky-300',     btn: 'bg-sky-600 hover:bg-sky-500',         dotOn: 'bg-sky-500',     dotPast: 'bg-sky-700' },
  };
  const ac = ACCENTS[accent] || ACCENTS.blue;

  return (
    <>
      <style>{`
        @keyframes tutor-pulse {
          0%, 100% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.62), 0 0 0 4px rgba(var(--tutor-rgb),0.35), 0 0 32px rgba(var(--tutor-rgb),0.4); }
          50% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.62), 0 0 0 6px rgba(var(--tutor-rgb),0.55), 0 0 44px rgba(var(--tutor-rgb),0.7); }
        }
        @keyframes tutor-fadein {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes tutor-fadein-center {
          from { opacity: 0; transform: translate(-50%, -50%) scale(0.96); }
          to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
        .tutor-tooltip { animation: tutor-fadein 0.18s ease-out both; }
        .tutor-tooltip-center { animation: tutor-fadein-center 0.22s ease-out both; }
        @keyframes tutor-content-in {
          from { opacity: 0; transform: translateX(5px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .tutor-content { animation: tutor-content-in 0.25s ease-out both; }
      `}</style>

      {/* Spotlight */}
      {box && !centered && (
        <div
          className="fixed pointer-events-none z-[150]"
          style={{
            top: box.top - 8, left: box.left - 8,
            width: box.width + 16, height: box.height + 16,
            borderRadius: 10,
            border: `2px solid rgb(${ac.rgb})`,
            ['--tutor-rgb']: ac.rgb,
            transition: 'top 0.28s cubic-bezier(0.4,0,0.2,1), left 0.28s cubic-bezier(0.4,0,0.2,1), width 0.28s cubic-bezier(0.4,0,0.2,1), height 0.28s cubic-bezier(0.4,0,0.2,1)',
            animation: 'tutor-pulse 1.6s ease-in-out infinite',
          }}
        />
      )}
      {centered && <div className="fixed inset-0 bg-black/65 z-[150] pointer-events-none" />}

      {/* Tooltip card */}
      <div
        className={`fixed z-[160] bg-slate-900 border-2 ${ac.card} rounded-xl shadow-2xl ${centered ? 'tutor-tooltip-center' : 'tutor-tooltip'}`}
        style={tipStyle}
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] font-mono text-slate-500 tracking-wide">
              STEP {step + 1} OF {steps.length}
            </div>
            <button onClick={onClose} className="text-slate-500 hover:text-slate-200 transition" aria-label="Close tour">
              <X size={14} />
            </button>
          </div>
          <div key={step} className="tutor-content">
            <h3 className={`text-base font-semibold ${ac.title} mb-2`}>{stepData.title}</h3>
            <p className="text-sm text-slate-300 leading-relaxed">{stepData.body}</p>
          </div>
          <div className="flex items-center gap-2 mt-4">
            <button onClick={onClose} className="text-xs text-slate-500 hover:text-slate-300 mr-auto">Skip tour</button>
            <button onClick={() => setStep(step - 1)} disabled={isFirst}
              className="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1">
              <ChevronLeft size={12} /> Back
            </button>
            <button onClick={() => isLast ? onClose() : setStep(step + 1)}
              className={`px-3 py-1.5 text-xs rounded ${ac.btn} text-white flex items-center gap-1 font-medium`}>
              {isLast ? 'Done' : <>Next <ChevronRight size={12} /></>}
            </button>
          </div>
          <div className="flex gap-0.5 mt-3">
            {steps.map((_, i) => (
              <button key={i} onClick={() => setStep(i)}
                className={`h-1 flex-1 rounded transition ${i === step ? ac.dotOn : i < step ? ac.dotPast : 'bg-slate-700 hover:bg-slate-600'}`}
                aria-label={`Go to step ${i + 1}`} />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ============ Guided-demo overlay ============
// Auto-advancing spotlight + narration for the hands-free demo. The App owns the
// step index and timing (so it can wait for background training); this component
// is purely presentational: it spotlights the target, shows the narration, and
// offers Pause/Resume and Stop. A progress bar fills over the step's dwell time.
function GuidedDemo({ step, steps, paused, waiting, onPause, onResume, onSkip, onStop }) {
  const [box, setBox] = useState(null);
  const tipRef = useRef(null);
  const stepData = steps[step];
  const target = stepData?.target;

  useEffect(() => {
    if (!target) { setBox(null); return; }
    let raf = 0, cancelled = false, last = null;
    const el0 = document.querySelector(target);
    if (el0) {
      const reduce = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el0.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center', inline: 'nearest' });
    }
    const tick = () => {
      if (cancelled) return;
      const el = document.querySelector(target);
      if (el) {
        const r = el.getBoundingClientRect();
        if (!last || Math.abs(r.top - last.top) > 0.5 || Math.abs(r.left - last.left) > 0.5 || Math.abs(r.width - last.width) > 0.5 || Math.abs(r.height - last.height) > 0.5) {
          last = { top: r.top, left: r.left, width: r.width, height: r.height };
          setBox(last);
        }
      } else if (last) { last = null; setBox(null); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelled = true; cancelAnimationFrame(raf); };
  }, [target]);

  // Keyboard: → or Space skip to next, Esc stops, P toggles pause. Ignored when
  // typing in a field.
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.key === 'ArrowRight' || e.key === ' ' || e.code === 'Space') { e.preventDefault(); onSkip(); }
      else if (e.key === 'Escape') { e.preventDefault(); onStop(); }
      else if (e.key.toLowerCase() === 'p') { e.preventDefault(); paused ? onResume() : onPause(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onSkip, onStop, onPause, onResume, paused]);

  // Modality for keyboard users: pointer events are blocked by the backdrop, but
  // Tab could still reach app controls behind it. Pull focus back into the demo
  // panel so only its own controls are reachable while it runs.
  useEffect(() => {
    const onFocusIn = (e) => {
      const panel = tipRef.current;
      if (!panel) return;
      if (!panel.contains(e.target)) {
        const first = panel.querySelector('button');
        if (first) first.focus();
      }
    };
    document.addEventListener('focusin', onFocusIn);
    // Move focus into the panel on mount so screen readers announce it.
    const t = setTimeout(() => { const p = tipRef.current; if (p) { const b = p.querySelector('button'); if (b) b.focus(); } }, 50);
    return () => { document.removeEventListener('focusin', onFocusIn); clearTimeout(t); };
  }, []);

  if (!stepData) return null;
  const centered = !target;
  const rgb = '167,139,250'; // violet accent, distinct from the manual tours
  const tipWidth = 360;
  let tipStyle = { width: tipWidth, transition: 'top 0.3s cubic-bezier(0.4,0,0.2,1), left 0.3s cubic-bezier(0.4,0,0.2,1), right 0.3s, bottom 0.3s' };
  if (centered) {
    tipStyle = { ...tipStyle, top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
  } else if (box) {
    const vw = window.innerWidth, vh = window.innerHeight;
    const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
    let pos = stepData.position || 'bottom';
    if (pos === 'right' && box.left + box.width + tipWidth + 32 > vw) pos = 'left';
    if (pos === 'left' && box.left - tipWidth - 32 < 0) pos = 'bottom';
    if (pos === 'top' && box.top < 240) pos = 'bottom';
    if (pos === 'bottom' && box.top + box.height + 240 > vh) pos = 'top';
    if (pos === 'top') tipStyle = { ...tipStyle, bottom: vh - box.top + 16, left: clamp(cx - tipWidth / 2, 12, vw - tipWidth - 12) };
    else if (pos === 'bottom') tipStyle = { ...tipStyle, top: box.top + box.height + 16, left: clamp(cx - tipWidth / 2, 12, vw - tipWidth - 12) };
    else if (pos === 'left') tipStyle = { ...tipStyle, right: vw - box.left + 16, top: clamp(cy - 110, 12, vh - 240) };
    else if (pos === 'right') tipStyle = { ...tipStyle, left: box.left + box.width + 16, top: clamp(cy - 110, 12, vh - 240) };
  } else {
    tipStyle = { ...tipStyle, top: 80, left: '50%', transform: 'translateX(-50%)' };
  }

  return (
    <>
      <style>{`
        @keyframes demo-pulse {
          0%,100% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.6), 0 0 0 4px rgba(${rgb},0.4), 0 0 34px rgba(${rgb},0.45); }
          50% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.6), 0 0 0 6px rgba(${rgb},0.6), 0 0 46px rgba(${rgb},0.75); }
        }
        @keyframes demo-progress { from { width: 0%; } to { width: 100%; } }
      `}</style>
      {/* Modal backdrop: blocks clicks reaching the app so the demo can't be
          desynced by stray interaction. Transparent when a spotlight is showing
          (the ring's huge box-shadow supplies the dimming); dimmed on centered
          steps. Wheel scrolling still works — pointer-events only affects
          hit-testing, not scroll propagation. */}
      <div
        className={`fixed inset-0 z-[149] ${centered ? 'bg-black/60' : ''}`}
        style={{ cursor: 'not-allowed' }}
        aria-hidden="true"
        title="The guided demo is running — press Stop to explore on your own"
        onClick={(e) => e.stopPropagation()}
      />
      {box && !centered && (
        <div className="fixed pointer-events-none z-[150]"
          style={{ top: box.top - 8, left: box.left - 8, width: box.width + 16, height: box.height + 16, borderRadius: 10,
            border: `2px solid rgb(${rgb})`, transition: 'top 0.3s cubic-bezier(0.4,0,0.2,1), left 0.3s, width 0.3s, height 0.3s', animation: 'demo-pulse 1.7s ease-in-out infinite' }} />
      )}

      <div ref={tipRef} role="dialog" aria-modal="true" aria-label={`Guided demo, step ${step + 1} of ${steps.length}`}
        className="fixed z-[160] bg-slate-900 border-2 border-violet-500 rounded-xl shadow-2xl" style={tipStyle}>
        {/* progress bar: a CSS keyframe that runs over the step's dwell; restarts
            per step via the key. Paused/waiting freeze it. */}
        <div className="h-1 rounded-t-xl bg-slate-800 overflow-hidden">
          <div key={`${step}-${paused}-${waiting}`} className="h-full bg-violet-500"
            style={(paused || waiting || !stepData.dwell) ? { width: waiting ? '100%' : '0%' } : { animation: `demo-progress ${stepData.dwell}ms linear forwards` }} />
        </div>
        <div className="p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] font-mono text-violet-300/80 tracking-wide flex items-center gap-1.5">
              <Play size={10} /> GUIDED DEMO · {step + 1}/{steps.length}
            </div>
            <button onClick={onStop} className="text-slate-500 hover:text-slate-200 transition" aria-label="Stop demo"><X size={14} /></button>
          </div>
          <h3 className="text-base font-semibold text-violet-200 mb-2">{stepData.title}</h3>
          <p className="text-sm text-slate-300 leading-relaxed">{stepData.body}</p>
          {waiting && (
            <div className="mt-2 flex items-center gap-1.5 text-[11px] text-violet-300/80">
              <Loader2 size={12} className="animate-spin" /> training…
            </div>
          )}
          <div className="flex items-center gap-2 mt-4">
            <button onClick={onStop} className="text-xs text-slate-500 hover:text-slate-300 mr-auto">Stop</button>
            {paused
              ? <button onClick={onResume} className="px-3 py-1.5 text-xs rounded bg-violet-600 hover:bg-violet-500 text-white flex items-center gap-1 font-medium"><Play size={12} /> Resume</button>
              : <button onClick={onPause} className="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1"><Pause size={12} /> Pause</button>}
            <button onClick={onSkip}
              className="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1 font-medium"
              title={step >= steps.length - 1 ? 'Finish the demo' : 'Skip to the next step'}>
              {step >= steps.length - 1 ? 'Finish' : <>Skip <SkipForward size={12} /></>}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

// ============ React App ============
export default function App() {
  const [algorithm, setAlgorithm] = useState('dqn');
  const [mode, setMode] = useState('beginner');
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(20);
  const [tick, setTick] = useState(0);
  const [grid, setGrid] = useState(PRESETS.hazards.grid);
  const [editMode, setEditMode] = useState(false);
  const [probeMode, setProbeMode] = useState(false);
  const [brush, setBrush] = useState(1);
  const [hp, setHp] = useState(DEFAULT_HP);
  const [bgStatus, setBgStatus] = useState(null);
  const [showOptimal, setShowOptimal] = useState(false);
  const [viewMode, setViewMode] = useState('q');
  const [inspectorTab, setInspectorTab] = useState('status'); // which learning-visual is shown next to the grid
  const [probes, setProbes] = useState([]);
  const [compareOn, setCompareOn] = useState(false);
  const [hpB, setHpB] = useState(DEFAULT_HP_B);
  const [narrationMode, setNarrationMode] = useState(false);
  const [palette, setPalette] = useState('default');
  // Sync the palette state to the module-level color setter synchronously during
  // render so all child components that call qColor/errorColor/accentBad/etc
  // in the SAME render pass see the new value. Doing this in useEffect would
  // cause a one-frame visual lag.
  setColorPalette(palette);
  const [slipProb, setSlipProb] = useState(0);
  const [rewardNoise, setRewardNoise] = useState(0);
  // Sync stochasticity to the Env's module-level config synchronously during
  // render, same rationale as the palette: every Env.step in this pass uses it.
  setEnvConfig({ slipProb, rewardNoise });
  const [tourStep, setTourStep] = useState(-1); // -1 = not active
  const [mathStep, setMathStep] = useState(-1); // -1 = not active
  const [techStep, setTechStep] = useState(-1); // -1 = not active; technical-panel tour
  const [demoStep, setDemoStep] = useState(-1);  // -1 = inactive; guided auto-demo index
  const [demoPaused, setDemoPaused] = useState(false);
  const demoTrainingRef = useRef(false);         // true while a demo step is waiting on bg training

  const envRef = useRef(null);
  const agentRef = useRef(null);
  const stateRef = useRef(null);
  const epRewardsRef = useRef([]);
  const lossesRef = useRef([]);
  const epsHistRef = useRef([]);
  const lastActionRef = useRef(null);
  const lastTransRef = useRef(null);
  const lastLearnRef = useRef(null);
  const currentEpRef = useRef({ ep: 0, step: 0, reward: 0 });
  const totalStepsRef = useRef(0);
  const syncFlashRef = useRef(0);
  const eventLogRef = useRef([]);
  const bgTrainingRef = useRef(false);
  const probeHistRef = useRef([]);
  const matchHistRef = useRef([]);
  const overestHistRef = useRef([]);   // per-episode: { ep, estA, estB, vstar } — agent max-Q(start) vs true V*(start)
  const probesRef = useRef([]);
  const visitsRef = useRef(new Uint32Array(W * Hg));
  // #3 Trajectory overlays: current episode's path of cells, plus a ring of the
  // most recent completed episodes (newest last) with their outcome.
  const curTrajRef = useRef([]);            // [{x,y}] for the in-progress episode
  const trajHistRef = useRef([]);           // [{ path:[{x,y}], outcome, ep }] recent episodes
  // #1 Value propagation: a snapshot of max-Q per cell at the start of the
  // current "window", and the per-cell delta (current − snapshot) we display.
  const vSnapRef = useRef(null);            // Float32Array(W*Hg) | null
  const vDeltaRef = useRef(new Float32Array(W * Hg));
  const MAX_TRAJ = 8;
  const narrationModeRef = useRef(false);
  const rolloutRef = useRef({ active: false, path: [], result: null, steps: 0, totalReward: 0 });
  const rolloutCancelRef = useRef(null);  // set while a rollout runs; aborts AND restores env state
  useEffect(() => { probesRef.current = probes; }, [probes]);
  useEffect(() => { narrationModeRef.current = narrationMode; }, [narrationMode]);

  const envBRef = useRef(null);
  const agentBRef = useRef(null);
  const stateBRef = useRef(null);
  const epRewardsBRef = useRef([]);
  const currentEpBRef = useRef({ ep: 0, step: 0, reward: 0 });
  const totalStepsBRef = useRef(0);

  const initAll = useCallback((newGrid, newHp, algo) => {
    envRef.current = new Env(newGrid);
    agentRef.current = makeAgent(algo, newHp);
    stateRef.current = envRef.current.reset();
    epRewardsRef.current = [];
    lossesRef.current = [];
    epsHistRef.current = [];
    lastActionRef.current = null;
    lastTransRef.current = null;
    lastLearnRef.current = null;
    currentEpRef.current = { ep: 0, step: 0, reward: 0 };
    totalStepsRef.current = 0;
    syncFlashRef.current = 0;
    eventLogRef.current = [{ t: 0, type: 'init', text: algo === 'tabular' ? 'Q-table initialized to zeros' : 'Network initialized with random weights' }];
    probeHistRef.current = [];
    matchHistRef.current = [];
    overestHistRef.current = [];
    curTrajRef.current = [];
    trajHistRef.current = [];
    vSnapRef.current = null;
    vDeltaRef.current = new Float32Array(W * Hg);
    visitsRef.current = new Uint32Array(W * Hg);
    visitsRef.current[envRef.current.sy * W + envRef.current.sx] = 1;
    rolloutRef.current = { active: false, path: [], result: null, steps: 0, totalReward: 0 };
    setTick(t => t + 1);
  }, []);

  const initB = useCallback((newGrid, newHpB, algo) => {
    envBRef.current = new Env(newGrid);
    agentBRef.current = makeAgent(algo, newHpB);
    stateBRef.current = envBRef.current.reset();
    epRewardsBRef.current = [];
    currentEpBRef.current = { ep: 0, step: 0, reward: 0 };
    totalStepsBRef.current = 0;
  }, []);

  useEffect(() => {
    if (skipAlgoInitRef.current) { skipAlgoInitRef.current = false; return; }
    initAll(grid, hp, algorithm);
    if (compareOn) initB(grid, hpB, algorithm);
    setRunning(false);
    setProbes([]);
    if (algorithm === 'tabular' && viewMode === 'tgtdiff') setViewMode('q');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm]);

  useEffect(() => {
    initAll(grid, hp, algorithm);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!agentRef.current) return;
    const cur = agentRef.current.hp;
    agentRef.current.hp = { ...cur, epsMin: hp.epsMin, epsDecay: hp.epsDecay, gamma: hp.gamma, alpha: hp.alpha, lr: hp.lr, batchSize: hp.batchSize, bufferSize: hp.bufferSize, targetUpdate: hp.targetUpdate, doubleDQN: hp.doubleDQN };
    if (agentRef.current.buffer.length > hp.bufferSize) agentRef.current.buffer = agentRef.current.buffer.slice(-hp.bufferSize);
  }, [hp]);

  useEffect(() => {
    if (!agentBRef.current) return;
    const cur = agentBRef.current.hp;
    agentBRef.current.hp = { ...cur, epsMin: hpB.epsMin, epsDecay: hpB.epsDecay, gamma: hpB.gamma, alpha: hpB.alpha, lr: hpB.lr, batchSize: hpB.batchSize, bufferSize: hpB.bufferSize, targetUpdate: hpB.targetUpdate, doubleDQN: hpB.doubleDQN };
  }, [hpB]);

  useEffect(() => {
    if (compareOn && !agentBRef.current) { initB(grid, hpB, algorithm); setTick(t => t + 1); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareOn]);

  const updateGrid = (g) => {
    setGrid(g);
    envRef.current = new Env(g);
    stateRef.current = envRef.current.reset();
    currentEpRef.current = { ...currentEpRef.current, step: 0, reward: 0 };
    lastTransRef.current = null;
    visitsRef.current = new Uint32Array(W * Hg);
    visitsRef.current[envRef.current.sy * W + envRef.current.sx] = 1;
    // Trajectories and value snapshots describe the OLD layout — drop them, or the
    // Paths view draws routes through cells that are now walls and the Flow view
    // diffs against a stale grid.
    curTrajRef.current = [];
    trajHistRef.current = [];
    vSnapRef.current = null;
    vDeltaRef.current = new Float32Array(W * Hg);
    rolloutRef.current = { active: false, path: [], result: null, steps: 0, totalReward: 0 };
    // Drop probes that the new layout turned into walls. Done here (rather than
    // only in the edit handler) so preset loads are covered too.
    setProbes(prev => {
      const next = prev.filter(p => g[p.y][p.x] !== 1);
      if (next.length !== prev.length) probeHistRef.current = [];
      return next;
    });
    if (compareOn) {
      envBRef.current = new Env(g);
      stateBRef.current = envBRef.current.reset();
      currentEpBRef.current = { ...currentEpBRef.current, step: 0, reward: 0 };
    }
    setTick(t => t + 1);
  };

  const doStep = useCallback(() => {
    const ag = agentRef.current, env = envRef.current;
    if (!ag || !env) return;
    const s = stateRef.current;
    const sx = env.x, sy = env.y;
    const action = ag.act(s);
    const { s: s2, r, terminal, truncated } = env.step(action.a);
    ag.remember(s, action.a, r, s2, terminal);
    const learn = ag.learn();
    lastActionRef.current = action;
    const transition = { sx, sy, dx: env.x, dy: env.y, action: action.a, reward: r, terminal };
    lastTransRef.current = transition;
    lastLearnRef.current = learn;
    currentEpRef.current.reward += r;
    currentEpRef.current.step++;
    totalStepsRef.current++;
    visitsRef.current[env.y * W + env.x]++;
    // #3: record the cell we just moved into for the trajectory overlay
    curTrajRef.current.push({ x: env.x, y: env.y });
    if (narrationModeRef.current) {
      eventLogRef.current.unshift({ t: totalStepsRef.current, type: 'narrate', text: buildNarration(action, transition, learn, algorithm, ag.eps, ag.buffer.length) });
    }
    if (learn?.sync) {
      syncFlashRef.current = 30;
      eventLogRef.current.unshift({ t: totalStepsRef.current, type: 'sync', text: `Target network synced (step ${totalStepsRef.current})` });
    } else if (syncFlashRef.current > 0) syncFlashRef.current--;
    if (learn?.loss != null) {
      lossesRef.current.push({ step: totalStepsRef.current, loss: learn.loss });
      if (lossesRef.current.length > 400) lossesRef.current.shift();
    }
    if (probesRef.current.length > 0 && totalStepsRef.current % 25 === 0) {
      const values = probesRef.current.map(p => {
        const f = ag.q.forward(env.encodeAt(p.x, p.y));
        return Math.max(f.q[0], f.q[1], f.q[2], f.q[3]);
      });
      probeHistRef.current.push({ step: totalStepsRef.current, values });
      if (probeHistRef.current.length > 200) probeHistRef.current.shift();
    }
    const epEnded = terminal || truncated;
    if (epEnded) {
      const reason = terminal && r === 1 ? 'goal' : terminal && r === -1 ? 'hazard' : 'timeout';
      epRewardsRef.current.push({ ep: currentEpRef.current.ep, reward: +currentEpRef.current.reward.toFixed(3), eps: +ag.eps.toFixed(3), steps: currentEpRef.current.step });
      epsHistRef.current.push({ ep: currentEpRef.current.ep, eps: +ag.eps.toFixed(4) });
      if (epRewardsRef.current.length > 300) epRewardsRef.current.shift();
      if (epsHistRef.current.length > 300) epsHistRef.current.shift();
      eventLogRef.current.unshift({ t: totalStepsRef.current, type: 'episode', text: `Ep ${currentEpRef.current.ep}: ${reason} (R=${currentEpRef.current.reward.toFixed(2)}, ${currentEpRef.current.step} steps)` });

      // #3: archive the completed episode's trajectory (start cell + visited path)
      const fullPath = [{ x: env.sx, y: env.sy }, ...curTrajRef.current];
      trajHistRef.current.push({ path: fullPath, outcome: reason, ep: currentEpRef.current.ep });
      if (trajHistRef.current.length > MAX_TRAJ) trajHistRef.current.shift();
      curTrajRef.current = [];

      // #1: snapshot max-Q per cell and compute the per-cell change since the last
      // snapshot. This is the "value wavefront" — where credit propagated this episode.
      const curV = readMaxQGrid(ag);
      if (vSnapRef.current) {
        const d = vDeltaRef.current;
        for (let i = 0; i < d.length; i++) d[i] = curV[i] - vSnapRef.current[i];
      }
      vSnapRef.current = curV;

      currentEpRef.current = { ep: currentEpRef.current.ep + 1, step: 0, reward: 0 };
      ag.decayEps();
      stateRef.current = env.reset();
      visitsRef.current[env.y * W + env.x]++;
    } else {
      stateRef.current = s2;
    }
    const cap = narrationModeRef.current ? 60 : 30;
    if (eventLogRef.current.length > cap) eventLogRef.current.length = cap;
  }, [algorithm]);

  const doStepB = useCallback(() => {
    const ag = agentBRef.current, env = envBRef.current;
    if (!ag || !env) return;
    const s = stateBRef.current;
    const action = ag.act(s);
    const { s: s2, r, terminal, truncated } = env.step(action.a);
    ag.remember(s, action.a, r, s2, terminal);
    ag.learn();
    currentEpBRef.current.reward += r;
    currentEpBRef.current.step++;
    totalStepsBRef.current++;
    if (terminal || truncated) {
      epRewardsBRef.current.push({ ep: currentEpBRef.current.ep, reward: +currentEpBRef.current.reward.toFixed(3), steps: currentEpBRef.current.step });
      if (epRewardsBRef.current.length > 300) epRewardsBRef.current.shift();
      currentEpBRef.current = { ep: currentEpBRef.current.ep + 1, step: 0, reward: 0 };
      ag.decayEps();
      stateBRef.current = env.reset();
    } else { stateBRef.current = s2; }
  }, []);

  const effectiveSpeed = narrationMode ? Math.min(speed, 2) : speed;

  useEffect(() => {
    if (!running) return;
    let raf, lastT = performance.now(), acc = 0;
    const loop = (now) => {
      const dt = Math.min(0.1, (now - lastT) / 1000);
      lastT = now;
      acc += dt * effectiveSpeed;
      let n = 0;
      while (acc >= 1 && n < 500) { doStep(); if (compareOn) doStepB(); acc -= 1; n++; }
      if (n > 0) setTick(t => t + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, effectiveSpeed, doStep, doStepB, compareOn]);

  const runBgTraining = useCallback((trainMode, target) => {
    if (!agentRef.current || bgTrainingRef.current) return;
    setRunning(false);
    bgTrainingRef.current = true;
    setBgStatus({ mode: trainMode, target });
    const startEp = currentEpRef.current.ep;
    let stepsDone = 0;
    const chunk = 400;
    const safetyMax = trainMode === 'episodes' ? target * (MAX_STEPS + 5) + 2000 : target;
    const isDone = () => {
      if (trainMode === 'steps') return stepsDone >= target;
      if (trainMode === 'episodes') return (currentEpRef.current.ep - startEp) >= target;
      return true;
    };
    const runChunk = () => {
      if (!bgTrainingRef.current) return;
      let n = 0;
      while (n < chunk && !isDone() && stepsDone < safetyMax) {
        doStep(); if (compareOn) doStepB(); stepsDone++; n++;
      }
      if (isDone() || stepsDone >= safetyMax) {
        bgTrainingRef.current = false;
        setBgStatus(null);
        setTick(t => t + 1);
      } else setTimeout(runChunk, 0);
    };
    setTimeout(runChunk, 0);
  }, [doStep, doStepB, compareOn]);

  const stepOnce = useCallback(() => {
    if (running || bgStatus || rolloutRef.current.active) return;
    doStep();
    if (compareOn) doStepB();
    setTick(t => t + 1);
  }, [running, bgStatus, doStep, doStepB, compareOn]);

  const stepEpisode = useCallback(() => {
    if (running || bgStatus || rolloutRef.current.active) return;
    const startEp = currentEpRef.current.ep;
    let safety = MAX_STEPS + 5;
    while (currentEpRef.current.ep === startEp && safety-- > 0) {
      doStep(); if (compareOn) doStepB();
    }
    setTick(t => t + 1);
  }, [running, bgStatus, doStep, doStepB, compareOn]);

  const runRollout = useCallback(() => {
    if (running || bgStatus || rolloutRef.current.active) return;
    const ag = agentRef.current, env = envRef.current;
    if (!ag || !env) return;
    const savedX = env.x, savedY = env.y, savedT = env.t;
    const savedLastAction = lastActionRef.current;
    const savedLastTrans = lastTransRef.current;
    env.reset();
    rolloutRef.current = { active: true, path: [{ x: env.x, y: env.y }], result: null, steps: 0, totalReward: 0 };
    setTick(t => t + 1);
    let restoreTimer = null;
    // Put the environment back exactly as it was before the rollout. A rollout is
    // a read-only demonstration, so training must resume from the same state.
    // If the env was replaced meanwhile (Reset / load session), skip the write —
    // otherwise we'd stamp stale coordinates onto a brand-new environment.
    const myEnv = env;
    const restore = () => {
      if (envRef.current === myEnv) {
        myEnv.x = savedX; myEnv.y = savedY; myEnv.t = savedT;
        lastActionRef.current = savedLastAction;
        lastTransRef.current = savedLastTrans;
      }
      rolloutRef.current = { active: false, path: [], result: null, steps: 0, totalReward: 0 };
      rolloutCancelRef.current = null;
      setTick(t => t + 1);
    };
    // External abort (demo skip/stop, reset). Without this, callers that just
    // flipped `active` would strand the env wherever the rollout walked to.
    rolloutCancelRef.current = () => {
      rolloutRef.current.active = false;
      if (restoreTimer) { clearTimeout(restoreTimer); restoreTimer = null; }
      restore();
    };
    const step = () => {
      const r = rolloutRef.current;
      if (!r.active) return;
      if (r.steps >= MAX_STEPS) { finish('timeout'); return; }
      const bestA = ag.actGreedy(env.encode());
      const { r: reward, terminal, truncated } = env.step(bestA);
      r.steps++;
      r.totalReward += reward;
      r.path.push({ x: env.x, y: env.y });
      lastActionRef.current = { a: bestA, explore: false };
      lastTransRef.current = null;
      setTick(t => t + 1);
      if (terminal) { finish(reward === 1 ? 'goal' : 'hazard'); return; }
      if (truncated) { finish('timeout'); return; }
      setTimeout(step, 220);
    };
    const finish = (result) => {
      rolloutRef.current.result = result;
      rolloutRef.current.active = false;
      setTick(t => t + 1);
      restoreTimer = setTimeout(() => { restoreTimer = null; restore(); }, 2800);
    };
    setTimeout(step, 220);
  }, [running, bgStatus]);

  const optimal = useMemo(() => valueIteration(grid, hp.gamma, slipProb), [grid, hp.gamma, slipProb]);

  const qMap = useMemo(() => {
    if (!agentRef.current || !envRef.current) return null;
    const out = [];
    for (let y = 0; y < Hg; y++) {
      const row = [];
      for (let x = 0; x < W; x++) {
        if (grid[y][x] === 1) { row.push(null); continue; }
        const f = agentRef.current.q.forward(envRef.current.encodeAt(x, y));
        row.push({ q: Array.from(f.q) });
      }
      out.push(row);
    }
    return { cells: out };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, grid, algorithm]);

  const tgtMap = useMemo(() => {
    if (viewMode !== 'tgtdiff' || algorithm !== 'dqn' || !agentRef.current || !envRef.current) return null;
    const out = [];
    for (let y = 0; y < Hg; y++) {
      const row = [];
      for (let x = 0; x < W; x++) {
        if (grid[y][x] === 1) { row.push(null); continue; }
        const f = agentRef.current.tgt.forward(envRef.current.encodeAt(x, y));
        row.push({ q: Array.from(f.q) });
      }
      out.push(row);
    }
    return { cells: out };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, grid, viewMode, algorithm]);

  const qScale = useMemo(() => {
    if (!qMap) return 0.01;
    let m = 0.01;
    for (const row of qMap.cells) for (const cell of row) {
      if (!cell) continue;
      for (const v of cell.q) if (Math.abs(v) > m) m = Math.abs(v);
    }
    return m;
  }, [qMap]);

  const vStarScale = useMemo(() => {
    let m = 0.01;
    for (let i = 0; i < optimal.V.length; i++) if (Math.abs(optimal.V[i]) > m) m = Math.abs(optimal.V[i]);
    return m;
  }, [optimal]);

  const tgtDiffScale = useMemo(() => {
    if (!qMap || !tgtMap) return 0.01;
    let m = 0.01;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const a = qMap.cells[y][x], b = tgtMap.cells[y][x];
      if (!a || !b) continue;
      const d = Math.abs(Math.max(...a.q) - Math.max(...b.q));
      if (d > m) m = d;
    }
    return m;
  }, [qMap, tgtMap]);

  const errorScale = useMemo(() => {
    if (!qMap) return 0.01;
    let m = 0.01;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const c = grid[y][x];
      if (c !== 0 && c !== 4) continue;
      const cell = qMap.cells[y][x];
      if (!cell) continue;
      const err = Math.abs(Math.max(...cell.q) - optimal.V[y * W + x]);
      if (err > m) m = err;
    }
    return m;
  }, [qMap, optimal, grid]);

  const maxVisits = useMemo(() => {
    let m = 1;
    for (let i = 0; i < visitsRef.current.length; i++) if (visitsRef.current[i] > m) m = visitsRef.current[i];
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const policyMatch = useMemo(() => {
    if (!qMap || !optimal) return null;
    let total = 0, match = 0;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const c = grid[y][x];
      if (c !== 0 && c !== 4) continue;
      const cell = qMap.cells[y][x];
      if (!cell) continue;
      let bestA = 0;
      for (let i = 1; i < 4; i++) if (cell.q[i] > cell.q[bestA]) bestA = i;
      total++;
      if (bestA === optimal.policy[y * W + x]) match++;
    }
    const sx = envRef.current?.sx, sy = envRef.current?.sy;
    const startV = sx != null ? optimal.V[sy * W + sx] : null;
    return { match, total, pct: total > 0 ? match / total : 0, startV };
  }, [qMap, optimal, grid]);

  // Derive a human-readable "where is the agent in its learning journey" phase
  // from real signals: how much it still explores (ε), how close its greedy
  // policy is to optimal, and whether recent episode rewards have stabilized.
  // This turns the abstract numbers into a narrative a learner can follow.
  const learningPhase = useMemo(() => {
    const steps = totalStepsRef.current;
    if (steps < 5) return { key: 'idle', label: 'Ready', color: 'slate', desc: 'Press Step or Animate to start training.' };
    const pct = policyMatch?.pct ?? 0;
    const rewards = epRewardsRef.current;
    const everReachedGoal = rewards.some(r => r.reward > 0.3);
    const recent = rewards.slice(-8).map(r => r.reward);
    let stable = false;
    if (recent.length >= 5) {
      const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
      const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
      stable = Math.sqrt(variance) < 0.25;
    }
    if (pct >= 0.999) return { key: 'converged', label: 'Converged', color: 'emerald', desc: 'The greedy policy matches optimal in every cell. Run a Rollout to watch it solve the grid.' };
    if (pct >= 0.85 && stable) return { key: 'near', label: 'Near-optimal', color: 'emerald', desc: 'Policy is mostly optimal and rewards have stabilized. A bit more training should finish it.' };
    if (pct >= 0.6 || (everReachedGoal && stable)) return { key: 'refining', label: 'Refining', color: 'sky', desc: 'The agent has found decent routes and is polishing details from less-visited cells.' };
    if (everReachedGoal) return { key: 'learning', label: 'Learning', color: 'amber', desc: 'It has reached the goal and value is propagating back along its paths. Keep training.' };
    return { key: 'exploring', label: 'Exploring', color: 'amber', desc: 'Still mostly random, mapping out the grid. Reaching the goal a few times kick-starts learning.' };
  }, [policyMatch, tick]);

  useEffect(() => {
    if (!policyMatch) return;
    const lastEp = epRewardsRef.current.length;
    const lastRecorded = matchHistRef.current[matchHistRef.current.length - 1];
    if (!lastRecorded || lastRecorded.ep !== lastEp) {
      matchHistRef.current.push({ ep: lastEp, pct: +(policyMatch.pct * 100).toFixed(1) });
      if (matchHistRef.current.length > 300) matchHistRef.current.shift();

      // Track Q-value overestimation: the agent's greedy max-Q at the start cell
      // vs the true optimal value V*(start). Vanilla DQN tends to sit above V*;
      // Double DQN hugs it. This is the clearest visual signature of the bias.
      const ag = agentRef.current, env = envRef.current;
      if (ag && env && optimal) {
        const f = ag.q.forward(env.encodeAt(env.sx, env.sy));
        const estA = Math.max(f.q[0], f.q[1], f.q[2], f.q[3]);
        const vstar = optimal.V[env.sy * W + env.sx];
        let estB = null;
        if (compareOn && agentBRef.current && envBRef.current) {
          const fb = agentBRef.current.q.forward(envBRef.current.encodeAt(envBRef.current.sx, envBRef.current.sy));
          estB = Math.max(fb.q[0], fb.q[1], fb.q[2], fb.q[3]);
        }
        overestHistRef.current.push({
          ep: lastEp,
          estA: +estA.toFixed(3),
          estB: estB == null ? null : +estB.toFixed(3),
          vstar: +vstar.toFixed(3),
        });
        if (overestHistRef.current.length > 300) overestHistRef.current.shift();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const policyMatchB = useMemo(() => {
    if (!compareOn || !agentBRef.current || !envBRef.current || !optimal) return null;
    let total = 0, match = 0;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const c = grid[y][x];
      if (c !== 0 && c !== 4) continue;
      const f = agentBRef.current.q.forward(envBRef.current.encodeAt(x, y));
      let bestA = 0;
      for (let i = 1; i < 4; i++) if (f.q[i] > f.q[bestA]) bestA = i;
      total++;
      if (bestA === optimal.policy[y * W + x]) match++;
    }
    return { match, total, pct: total > 0 ? match / total : 0 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, grid, optimal, compareOn]);

  const handleCellClick = (x, y) => {
    if (bgStatus || rolloutRef.current.active) return;
    const cellType = grid[y][x];
    if (probeMode) {
      if (cellType === 1) return;
      setProbes(prev => {
        const i = prev.findIndex(p => p.x === x && p.y === y);
        if (i >= 0) { const next = prev.filter((_, j) => j !== i); probeHistRef.current = []; return next; }
        if (prev.length >= 4) return prev;
        const used = new Set(prev.map(p => p.color));
        const color = PROBE_COLORS.find(c => !used.has(c)) || PROBE_COLORS[0];
        probeHistRef.current = [];
        return [...prev, { x, y, color }];
      });
      return;
    }
    if (editMode) {
      const ng = grid.map(r => [...r]);
      if (brush === 4) for (let yy = 0; yy < Hg; yy++) for (let xx = 0; xx < W; xx++) if (ng[yy][xx] === 4) ng[yy][xx] = 0;
      ng[y][x] = brush;
      updateGrid(ng);   // also drops probes that became walls
    }
  };

  const reset = () => {
    if (bgStatus) return;
    rolloutRef.current.active = false;
    initAll(grid, hp, algorithm);
    if (compareOn) initB(grid, hpB, algorithm);
    setRunning(false);
    setProbes([]);
  };
  const resetB = () => {
    if (bgStatus) return;
    initB(grid, hpB, algorithm);
    setTick(t => t + 1);
  };

  // Confirm-guard for destructive actions. We only prompt once meaningful
  // training exists (so early clicks aren't nagged); below the threshold the
  // action runs immediately. The pending action is stored as a thunk.
  const PROGRESS_THRESHOLD = 200; // steps
  const [pendingConfirm, setPendingConfirm] = useState(null); // { title, body, confirmLabel, run } | null
  const hasProgress = () => totalStepsRef.current >= PROGRESS_THRESHOLD;
  const guardDestructive = (cfg, run) => {
    if (bgStatus) return;
    if (hasProgress()) setPendingConfirm({ ...cfg, run });
    else run();
  };
  const stepsLabel = () => totalStepsRef.current.toLocaleString();

  const requestReset = () => guardDestructive({
    title: 'Reset the agent?',
    body: `This clears everything the agent has learned over ${stepsLabel()} steps and starts training from scratch. This can't be undone.`,
    confirmLabel: 'Reset',
  }, reset);

  const requestResetB = () => {
    if (bgStatus) return;
    if (totalStepsBRef.current >= PROGRESS_THRESHOLD) {
      setPendingConfirm({
        title: 'Reset agent B?',
        body: `This rebuilds agent B and discards its ${totalStepsBRef.current.toLocaleString()} steps of learning. This can't be undone.`,
        confirmLabel: 'Reset B',
        run: resetB,
      });
    } else { resetB(); }
  };

  const requestPreset = (g) => guardDestructive({
    title: 'Load this preset?',
    body: `Switching the grid layout discards the agent's current progress (${stepsLabel()} steps of learning), since its values no longer apply to the new map.`,
    confirmLabel: 'Load preset',
  }, () => updateGrid(g));

  const startTour = () => setTourStep(0);
  const closeTour = () => setTourStep(-1);
  const startMath = () => setMathStep(0);
  const closeMath = () => setMathStep(-1);
  const startTech = () => setTechStep(0);
  const closeTech = () => setTechStep(-1);

  // ===== Guided auto-demo driver =====
  // Starting the demo closes any manual overlay and resets to a clean agent so the
  // walkthrough always begins from the same blank state.
  const startDemo = () => {
    setTourStep(-1); setTechStep(-1); setMathStep(-1); setPendingConfirm(null);
    setRunning(false);
    setCompareOn(false);
    setViewMode('q'); setInspectorTab('status');
    setDemoPaused(false);
    // Clear any work still in flight from a previous demo or a manual fast-train,
    // otherwise runBgTraining would early-return and the demo's training step
    // would be silently skipped.
    demoTrainingRef.current = false;
    if (bgTrainingRef.current) { bgTrainingRef.current = false; setBgStatus(null); }
    if (rolloutCancelRef.current) rolloutCancelRef.current();
    // Force step 0's action to run even if the demo was already sitting on step 0.
    demoExecutedRef.current = -1;
    setDemoStep(0);
  };
  const stopDemo = useCallback(() => {
    setDemoStep(-1);
    setDemoPaused(false);
    demoTrainingRef.current = false;
    setRunning(false);
    if (bgTrainingRef.current) { bgTrainingRef.current = false; setBgStatus(null); }
    if (rolloutCancelRef.current) rolloutCancelRef.current();
  }, []);
  const advanceDemo = useCallback(() => {
    setDemoStep(s => {
      if (s < 0) return s;
      if (s + 1 >= GUIDED_DEMO.length) { demoTrainingRef.current = false; return -1; }
      return s + 1;
    });
  }, []);
  // Skip to the next step immediately. If the current step is mid-training, halt
  // the background run first and clear the wait flag so its completion-watch won't
  // fire a second advance; also cancel any active rollout. Then advance.
  const skipDemo = useCallback(() => {
    demoTrainingRef.current = false;
    if (bgTrainingRef.current) { bgTrainingRef.current = false; setBgStatus(null); }
    if (rolloutCancelRef.current) rolloutCancelRef.current();
    advanceDemo();
  }, [advanceDemo]);

  // Execute each demo step's action exactly once on entry (guarded by a ref so
  // pause/resume doesn't re-run it). View/tab are set on entry too.
  const demoExecutedRef = useRef(-1);
  useEffect(() => {
    if (demoStep < 0 || demoStep >= GUIDED_DEMO.length) { demoExecutedRef.current = -1; return; }
    if (demoExecutedRef.current === demoStep) return; // already ran this step's action
    demoExecutedRef.current = demoStep;
    const step = GUIDED_DEMO[demoStep];
    if (step.viewMode) setViewMode(step.viewMode);
    if (step.tab) setInspectorTab(step.tab);
    const act = step.action;
    if (act) {
      if (act.type === 'reset') { initAll(grid, hp, algorithm); }
      else if (act.type === 'steps') { for (let i = 0; i < act.n; i++) { doStep(); if (compareOn) doStepB(); } setTick(t => t + 1); }
      else if (act.type === 'train') { demoTrainingRef.current = true; runBgTraining('steps', act.steps); }
      else if (act.type === 'rollout') { runRollout(); }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoStep]);

  // Schedule auto-advance for read-only steps. Re-runs on pause/resume (so resume
  // restarts the dwell) but never re-runs the action. Training steps are advanced
  // by the bgStatus-watch effect instead.
  useEffect(() => {
    if (demoStep < 0 || demoStep >= GUIDED_DEMO.length || demoPaused) return;
    const step = GUIDED_DEMO[demoStep];
    if (step.action?.type === 'train') return; // advanced on training completion
    const t = setTimeout(advanceDemo, step.dwell ?? 6000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoStep, demoPaused]);

  // When a demo training step finishes (bgStatus returns to null), linger briefly
  // on the result, then advance.
  useEffect(() => {
    if (demoStep < 0 || demoPaused) return;
    if (demoTrainingRef.current && !bgStatus) {
      demoTrainingRef.current = false;
      const t = setTimeout(advanceDemo, 1600);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bgStatus, demoStep, demoPaused]);

  // ===== Session persistence (artifact key-value storage) =====
  const SAVE_PREFIX = 'dqnviz:save:';
  const [saves, setSaves] = useState([]);            // [{ key, name, savedAt, algorithm, totalSteps }]
  const [saveStatus, setSaveStatus] = useState(null); // transient toast: { kind, text }
  const [storageReady, setStorageReady] = useState(typeof window !== 'undefined' && !!window.storage);
  // When a load changes the algorithm, the [algorithm] effect would normally
  // re-init (wiping the agent we just loaded). This ref tells that effect to
  // skip exactly once.
  const skipAlgoInitRef = useRef(false);

  const flash = (kind, text) => { setSaveStatus({ kind, text }); setTimeout(() => setSaveStatus(null), 2600); };

  // Storage keys may not contain whitespace, path separators or quotes, but save
  // names are free text (and the default name is a locale timestamp full of
  // spaces and slashes). Derive a sanitized key from the name; it's deterministic,
  // so re-saving under the same name overwrites that entry as users expect. The
  // human-readable name is kept inside the payload and used for display.
  const keyFor = (name) => {
    const slug = String(name).trim()
      .replace(/[^A-Za-z0-9._-]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 120);
    return SAVE_PREFIX + (slug || 'session');
  };

  const refreshSaves = useCallback(async () => {
    if (!window.storage) { setStorageReady(false); return; }
    try {
      const res = await window.storage.list(SAVE_PREFIX, false);
      const keys = res?.keys || [];
      const metas = [];
      for (const k of keys) {
        try {
          const r = await window.storage.get(k, false);
          if (r?.value) { const o = JSON.parse(r.value); metas.push({ key: k, name: o.name, savedAt: o.savedAt, algorithm: o.algorithm, totalSteps: o.totalSteps }); }
        } catch { /* skip unreadable entry */ }
      }
      metas.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
      setSaves(metas);
      setStorageReady(true);
    } catch { setStorageReady(false); }
  }, []);

  useEffect(() => { refreshSaves(); }, [refreshSaves]);

  const buildSnapshot = (name) => ({
    v: 1,
    name,
    savedAt: Date.now(),
    algorithm,
    totalSteps: totalStepsRef.current,
    grid,
    hp,
    viewMode,
    showOptimal,
    palette,
    stochastic: { slipProb, rewardNoise },
    agent: agentRef.current ? agentRef.current.serialize() : null,
    history: {
      epRewards: epRewardsRef.current.slice(-300),
      losses: lossesRef.current.slice(-300),
      epsHist: epsHistRef.current.slice(-300),
      matchHist: matchHistRef.current.slice(-300),
      overestHist: overestHistRef.current.slice(-300),
    },
    episode: { ep: currentEpRef.current.ep },
  });

  const saveSession = async (rawName) => {
    if (!window.storage) { flash('error', 'Storage unavailable in this environment'); return; }
    if (bgStatus) { flash('error', 'Wait for training to finish'); return; }
    const name = (rawName || '').trim() || `Save ${new Date().toLocaleString()}`;
    try {
      const snap = buildSnapshot(name);
      const res = await window.storage.set(keyFor(name), JSON.stringify(snap), false);
      if (!res) { flash('error', 'Save failed'); return; }
      flash('ok', `Saved “${name}”`);
      refreshSaves();
    } catch (e) {
      const msg = String(e);
      flash('error', /5MB|size|large/i.test(msg) ? 'Save too large — reduce replay capacity' : 'Save failed');
    }
  };

  const applySnapshot = (snap) => {
    const algo = snap.algorithm || 'dqn';
    if (algo !== algorithm) { skipAlgoInitRef.current = true; setAlgorithm(algo); }
    setGrid(snap.grid);
    // Merge over defaults rather than replacing: a session saved before a newer
    // hyperparameter existed would otherwise leave that field undefined and feed
    // `undefined` to controlled inputs.
    setHp({ ...DEFAULT_HP, ...(snap.hp || {}) });
    setViewMode(snap.viewMode || 'q');
    setShowOptimal(!!snap.showOptimal);
    if (snap.palette) setPalette(snap.palette);
    if (snap.stochastic) { setSlipProb(snap.stochastic.slipProb || 0); setRewardNoise(snap.stochastic.rewardNoise || 0); }

    envRef.current = new Env(snap.grid);
    agentRef.current = snap.agent ? deserializeAgent(snap.agent) : makeAgent(algo, { ...DEFAULT_HP, ...(snap.hp || {}) });
    stateRef.current = envRef.current.reset();
    const h = snap.history || {};
    epRewardsRef.current = h.epRewards || [];
    lossesRef.current = h.losses || [];
    epsHistRef.current = h.epsHist || [];
    matchHistRef.current = h.matchHist || [];
    overestHistRef.current = h.overestHist || [];
    lastActionRef.current = null;
    lastTransRef.current = null;
    lastLearnRef.current = null;
    currentEpRef.current = { ep: snap.episode?.ep || 0, step: 0, reward: 0 };
    totalStepsRef.current = snap.totalSteps || 0;
    syncFlashRef.current = 0;
    eventLogRef.current = [{ t: totalStepsRef.current, type: 'init', text: `Loaded session “${snap.name}” (${(snap.totalSteps || 0).toLocaleString()} steps)` }];
    probeHistRef.current = [];
    curTrajRef.current = [];
    trajHistRef.current = [];
    vSnapRef.current = readMaxQGrid(agentRef.current);
    vDeltaRef.current = new Float32Array(W * Hg);
    visitsRef.current = new Uint32Array(W * Hg);
    visitsRef.current[envRef.current.sy * W + envRef.current.sx] = 1;
    rolloutRef.current = { active: false, path: [], result: null, steps: 0, totalReward: 0 };
    setRunning(false);
    setProbes([]);
    setTick(t => t + 1);
  };

  const loadSession = async (key) => {
    if (!window.storage) { flash('error', 'Storage unavailable'); return; }
    if (bgStatus) { flash('error', 'Wait for training to finish'); return; }
    try {
      const r = await window.storage.get(key, false);
      if (!r?.value) { flash('error', 'Could not read save'); return; }
      const snap = JSON.parse(r.value);
      const doLoad = () => { applySnapshot(snap); flash('ok', `Loaded “${snap.name}”`); };
      if (hasProgress()) {
        setPendingConfirm({
          title: 'Load this session?',
          body: `This replaces the current agent (${stepsLabel()} steps) with the saved one. Unsaved progress will be lost.`,
          confirmLabel: 'Load',
          run: doLoad,
        });
      } else { doLoad(); }
    } catch { flash('error', 'Save file is corrupted'); }
  };

  const deleteSession = async (key, name) => {
    if (!window.storage) return;
    try {
      await window.storage.delete(key, false);
      flash('ok', `Deleted “${name}”`);
      refreshSaves();
    } catch { flash('error', 'Delete failed'); }
  };

  // Export per-episode training data as CSV via a Blob download. Pulls from the
  // same history the charts use, joining episode reward / steps / ε / policy-match.
  const exportCsv = () => {
    const rewards = epRewardsRef.current;
    if (!rewards.length) { flash('error', 'No episodes to export yet'); return; }
    const matchByEp = new Map(matchHistRef.current.map(m => [m.ep, m.pct]));
    const rows = [['episode', 'reward', 'steps', 'epsilon', 'policy_match_pct']];
    rewards.forEach((r, i) => {
      const pm = matchByEp.has(i + 1) ? matchByEp.get(i + 1) : '';
      rows.push([r.ep, r.reward, r.steps, r.eps, pm]);
    });
    const csv = rows.map(row => row.join(',')).join('\n');
    try {
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dqn-training-${algorithm}-${rewards.length}ep.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      flash('ok', `Exported ${rewards.length} episodes`);
    } catch { flash('error', 'Export failed'); }
  };

  const isTech = mode === 'technical';
  const isTabular = algorithm === 'tabular';
  const techSteps = isTabular ? TABULAR_TECH_TOUR : DQN_TECH_TOUR;
  const techAccent = isTabular ? 'emerald' : 'sky';

  // If the user switches mode/algorithm mid-tour, the targeted panels can vanish;
  // close the technical tour so we don't strand an orphaned spotlight.
  useEffect(() => {
    if (techStep >= 0 && !isTech) setTechStep(-1);
  }, [isTech, techStep]);
  // Switching algorithm swaps which tour applies; restart it from the top so the
  // step content matches the panels now on screen.
  useEffect(() => {
    if (techStep >= 0) setTechStep(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm]);

  // Keyboard shortcuts for the simulation: Space = Animate/Pause, S = Step,
  // E = Episode, R = Rollout. Suppressed while typing in a field or while any
  // tour / lesson overlay is open (those own their own keys).
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (tourStep >= 0 || techStep >= 0 || mathStep >= 0 || demoStep >= 0 || pendingConfirm) return;
      const busy = !!bgStatus || rolloutRef.current.active;
      if (e.key === ' ' || e.code === 'Space') {
        if (busy) return;
        e.preventDefault();
        setRunning(r => !r);
        return;
      }
      const k = e.key.toLowerCase();
      if (k === 's') { e.preventDefault(); stepOnce(); }
      else if (k === 'e') { e.preventDefault(); stepEpisode(); }
      else if (k === 'r') { e.preventDefault(); runRollout(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bgStatus, stepOnce, stepEpisode, runRollout, tourStep, techStep, mathStep, demoStep, pendingConfirm]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4" style={{ fontFamily: 'ui-sans-serif, system-ui' }}>
      {/* Respect prefers-reduced-motion: neutralize looping/entrance animations and
          long transitions for users with vestibular sensitivity. Spinners keep a
          slow rotation so "busy" is still legible; everything else snaps. */}
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
          }
          .animate-spin { animation: spin 1.4s linear infinite !important; }
        }
      `}</style>
      <div className="max-w-7xl mx-auto">
        <Header
          algorithm={algorithm} setAlgorithm={setAlgorithm}
          mode={mode} setMode={setMode}
          compareOn={compareOn} setCompareOn={setCompareOn}
          onStartTour={startTour}
          onStartMath={startMath}
          onStartDemo={startDemo}
          palette={palette} setPalette={setPalette}
        />

        <div className="flex gap-4 mt-4 items-start">
            <div className="space-y-4 min-w-0 sticky top-4 self-start shrink-0" style={{ width: 'clamp(220px, 38%, 456px)' }}>
              <GridPanel
              algorithm={algorithm}
              grid={grid} qMap={qMap} qScale={qScale} env={envRef.current}
              lastTrans={lastTransRef.current} lastAction={lastActionRef.current}
              editMode={editMode} setEditMode={setEditMode}
              probeMode={probeMode} setProbeMode={setProbeMode}
              probes={probes}
              brush={brush} setBrush={setBrush}
              onCellClick={handleCellClick}
              syncFlash={syncFlashRef.current}
              bgStatus={bgStatus}
              optimal={optimal} showOptimal={showOptimal} setShowOptimal={setShowOptimal}
              viewMode={viewMode} setViewMode={setViewMode}
              tgtMap={tgtMap} vStarScale={vStarScale} tgtDiffScale={tgtDiffScale}
              errorScale={errorScale} maxVisits={maxVisits}
              visits={visitsRef.current}
              rollout={rolloutRef.current}
              trajectories={trajHistRef.current}
              vDelta={vDeltaRef.current}
            />
            <ControlPanel
              running={running} setRunning={setRunning}
              speed={speed} setSpeed={setSpeed}
              effectiveSpeed={effectiveSpeed}
              onReset={requestReset}
              onStepOnce={stepOnce} onStepEpisode={stepEpisode}
              onFastSteps={() => runBgTraining('steps', 5000)}
              onFastEpisodes={(n) => runBgTraining('episodes', n)}
              onRollout={runRollout}
              rolloutActive={rolloutRef.current.active}
              narrationMode={narrationMode} setNarrationMode={setNarrationMode}
              bgStatus={bgStatus}
              presets={PRESETS} updateGrid={requestPreset}
            />
          </div>

          <div className="flex-1 min-w-0">
            <InspectorPanel
              tab={inspectorTab} setTab={setInspectorTab}
              algorithm={algorithm} isTech={isTech} isTabular={isTabular} mode={mode}
              compareOn={compareOn}
              narrationMode={narrationMode}
              onStartTech={startTech}
              // status
              currentEp={currentEpRef.current} agent={agentRef.current}
              totalSteps={totalStepsRef.current}
              lastAction={lastActionRef.current} lastLearn={lastLearnRef.current}
              policyMatch={policyMatch} phase={learningPhase}
              hp={hp} setHp={setHp}
              // bellman / probes
              probes={probes} probeHist={probeHistRef.current} optimalV={optimal.V}
              // model
              state={stateRef.current} syncFlash={syncFlashRef.current}
              grid={grid} qMap={qMap} optimalPolicy={optimal.policy}
              // charts
              epRewards={epRewardsRef.current} epRewardsB={compareOn ? epRewardsBRef.current : null}
              matchHist={matchHistRef.current} overestHist={overestHistRef.current}
              losses={lossesRef.current} epsHist={epsHistRef.current}
              doubleA={hp.doubleDQN} doubleB={hpB.doubleDQN}
              // memory
              buffer={agentRef.current?.buffer ?? []}
              // settings
              slipProb={slipProb} setSlipProb={setSlipProb}
              rewardNoise={rewardNoise} setRewardNoise={setRewardNoise}
              archChanged={!isTabular && !!agentRef.current && archDiffers(hp, agentRef.current.q)}
              // log / save
              eventLog={eventLogRef.current}
              saves={saves} storageReady={storageReady} saveStatus={saveStatus}
              onSave={saveSession} onLoad={loadSession} onDelete={deleteSession} onExport={exportCsv}
              bgStatus={bgStatus}
            />
          </div>
        </div>

        {compareOn && (
          <ComparePanel
            algorithm={algorithm}
            hpB={hpB} setHpB={setHpB} onReset={requestResetB}
            policyMatch={policyMatch} policyMatchB={policyMatchB}
            agentB={agentBRef.current} epRewardsB={epRewardsBRef.current}
            grid={grid} qMapA={qMap} agentBObj={agentBRef.current} env={envRef.current}
            optimalPolicy={optimal.policy} tick={tick}
            disabled={!!bgStatus}
          />
        )}

        {!isTech && <BeginnerExplain algorithm={algorithm} onStartTour={startTour} onStartMath={startMath} />}

        <Footer algorithm={algorithm} doubleDQN={!isTabular && hp.doubleDQN} />
      </div>

      {tourStep >= 0 && (
        <Tutorial step={tourStep} setStep={setTourStep} onClose={closeTour}
          onStepEnter={(sd) => { if (sd.tab) setInspectorTab(sd.tab); }} />
      )}
      {techStep >= 0 && (
        <Tutorial step={techStep} setStep={setTechStep} onClose={closeTech} steps={techSteps} accent={techAccent}
          onStepEnter={(sd) => { if (sd.tab) setInspectorTab(sd.tab); }} />
      )}
      {demoStep >= 0 && (
        <GuidedDemo
          step={demoStep} steps={GUIDED_DEMO}
          paused={demoPaused} waiting={!!bgStatus && demoTrainingRef.current}
          onPause={() => setDemoPaused(true)}
          onResume={() => setDemoPaused(false)}
          onSkip={skipDemo}
          onStop={stopDemo}
        />
      )}
      {mathStep >= 0 && (
        <MathLesson step={mathStep} setStep={setMathStep} onClose={closeMath} />
      )}
      {pendingConfirm && (
        <ConfirmDialog
          title={pendingConfirm.title}
          body={pendingConfirm.body}
          confirmLabel={pendingConfirm.confirmLabel}
          onConfirm={() => { pendingConfirm.run(); setPendingConfirm(null); }}
          onCancel={() => setPendingConfirm(null)}
        />
      )}
    </div>
  );
}

// ============ Info tooltip (?) ============
function Info({ text, w = 240 }) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const ref = useRef(null);
  const handleEnter = () => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setPos({ top: rect.top - 6, left: rect.left + rect.width / 2 });
    }
    setShow(true);
  };
  return (
    <span ref={ref} className="relative inline-flex items-center cursor-help align-middle ml-1"
      onMouseEnter={handleEnter} onMouseLeave={() => setShow(false)}>
      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-[9px] bg-slate-700 text-slate-300 hover:bg-slate-600 font-bold leading-none">?</span>
      {show && (
        <span className="block fixed z-[100] px-2.5 py-1.5 bg-slate-800 border border-slate-600 rounded text-[11px] text-slate-100 leading-snug shadow-2xl pointer-events-none normal-case font-normal"
          style={{ width: w, top: pos.top, left: pos.left, transform: 'translate(-50%, -100%)' }}>
          {text}
        </span>
      )}
    </span>
  );
}

function Header({ algorithm, setAlgorithm, mode, setMode, compareOn, setCompareOn, onStartTour, onStartMath, onStartDemo, palette, setPalette }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 flex-wrap">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2 flex-wrap">
          Q-Learning Visualizer
          <button onClick={onStartDemo}
            className="ml-1 px-2.5 py-1 rounded text-xs font-semibold bg-violet-600 text-white hover:bg-violet-500 transition flex items-center gap-1 shadow-sm"
            title="Sit back and watch an automatic guided run from a blank agent to a solved grid">
            <Play size={12} /> Watch demo
          </button>
          <button onClick={onStartTour}
            className="px-2 py-1 rounded text-xs font-medium bg-blue-500/15 border border-blue-500/40 text-blue-300 hover:bg-blue-500/25 hover:border-blue-500/70 transition flex items-center gap-1"
            title="Start the interactive tutorial">
            Take the tour
          </button>
          <button onClick={onStartMath}
            className="px-2 py-1 rounded text-xs font-medium bg-violet-500/15 border border-violet-500/40 text-violet-300 hover:bg-violet-500/25 hover:border-violet-500/70 transition flex items-center gap-1"
            title="Walk through the mathematics of reinforcement learning">
            The math
          </button>
        </h1>
        <p className="text-sm text-slate-400">Watch a {algorithm === 'tabular' ? 'tabular Q-learning' : 'DQN'} agent learn to navigate a grid world.</p>
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <div data-tutor="algo-toggle" className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
          <button onClick={() => setAlgorithm('tabular')}
            className={`px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1.5 transition ${
              algorithm === 'tabular' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}>
            <Grid3x3 size={14} /> Tabular Q
          </button>
          <button onClick={() => setAlgorithm('dqn')}
            className={`px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1.5 transition ${
              algorithm === 'dqn' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}>
            <Network size={14} /> Deep Q
          </button>
        </div>
        <button onClick={() => setCompareOn(!compareOn)}
          className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 border transition ${
            compareOn ? 'bg-fuchsia-600/20 border-fuchsia-500 text-fuchsia-200' : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}>
          <GitCompare size={14} /> Compare A/B
        </button>
        <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
          <button onClick={() => setMode('beginner')}
            className={`px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1.5 transition ${
              mode === 'beginner' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}>
            <BookOpen size={14} /> Beginner
          </button>
          <button onClick={() => setMode('technical')}
            className={`px-3 py-1.5 rounded text-sm font-medium flex items-center gap-1.5 transition ${
              mode === 'technical' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}>
            <Cpu size={14} /> Technical
          </button>
        </div>
        <button onClick={() => setPalette(palette === 'cbsafe' ? 'default' : 'cbsafe')}
          title={palette === 'cbsafe'
            ? 'Switch to default red/green palette'
            : 'Switch to a colorblind-safe orange/blue palette'}
          aria-label="Toggle colorblind-safe palette"
          className={`p-1.5 rounded-lg border transition flex items-center justify-center ${
            palette === 'cbsafe'
              ? 'bg-amber-600/20 border-amber-500 text-amber-300'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}>
          <Eye size={16} />
        </button>
      </div>
    </div>
  );
}

function Cell({ cellType, qVals, isAgent, isUpdSrc, isUpdDst, qScale, lastAction, editMode, probeMode, onClick, size,
  optimalAction, showOptimal, viewMode, vStar, vStarScale, tgtMaxQ, tgtDiffScale, errorVal, errorScale, visitCount, maxVisits, vDeltaVal, vDeltaScale, probeColor }) {
  const c = size / 2;
  let mq = -Infinity, minQ = Infinity, bestIdx = 0;
  for (let i = 0; i < 4; i++) { if (qVals[i] > mq) { mq = qVals[i]; bestIdx = i; } if (qVals[i] < minQ) minQ = qVals[i]; }
  const allEq = mq - minQ < 1e-6;
  const disagrees = showOptimal && optimalAction != null && optimalAction !== bestIdx && !allEq;
  let stroke = '#cbd5e1', sw = 1;
  if (isUpdSrc) { stroke = '#f59e0b'; sw = 3; }
  else if (isUpdDst) { stroke = '#3b82f6'; sw = 2.5; }
  else if (disagrees) { stroke = accentBad(); sw = 2.5; }
  const greedyColor = disagrees ? accentBad() : (showOptimal && !allEq && optimalAction === bestIdx ? accentGood() : '#0f172a');
  const cur = (editMode || probeMode) ? 'pointer' : 'default';

  if (cellType === 1) return (
    <g onClick={onClick} style={{ cursor: cur }}>
      <rect width={size} height={size} fill="#475569" />
      <line x1="0" y1="0" x2={size} y2={size} stroke="#1e293b" strokeWidth="1.5" />
      <line x1={size} y1="0" x2="0" y2={size} stroke="#1e293b" strokeWidth="1.5" />
      <rect x={sw/2} y={sw/2} width={size-sw} height={size-sw} fill="none" stroke="#cbd5e1" strokeWidth="1" />
    </g>
  );
  if (cellType === 3) return (
    <g onClick={onClick} style={{ cursor: cur }}>
      <rect width={size} height={size} fill="#7f1d1d" />
      <text x={c} y={c+9} textAnchor="middle" fontSize="26">💀</text>
      <rect x={sw/2} y={sw/2} width={size-sw} height={size-sw} fill="none" stroke="#cbd5e1" strokeWidth="1" />
    </g>
  );
  if (cellType === 2) return (
    <g onClick={onClick} style={{ cursor: cur }}>
      <rect width={size} height={size} fill="#a16207" />
      <text x={c} y={c+9} textAnchor="middle" fontSize="26">🚩</text>
      <rect x={sw/2} y={sw/2} width={size-sw} height={size-sw} fill="none" stroke="#cbd5e1" strokeWidth="1" />
    </g>
  );

  if (viewMode === 'q') {
    const tris = [`0,0 ${size},0 ${c},${c}`, `${size},0 ${size},${size} ${c},${c}`, `${size},${size} 0,${size} ${c},${c}`, `0,${size} 0,0 ${c},${c}`];
    return (
      <g onClick={onClick} style={{ cursor: cur }}>
        {tris.map((pts, i) => <polygon key={i} points={pts} fill={qColor(qVals[i], qScale)} stroke="rgba(0,0,0,0.06)" strokeWidth="0.5" />)}
        <g fontFamily="ui-monospace, Menlo, monospace" fontSize="9" fill="#1f2937" pointerEvents="none">
          <text x={c} y={size*0.27} textAnchor="middle">{qVals[0].toFixed(2)}</text>
          <text x={size*0.74} y={c+3} textAnchor="middle">{qVals[1].toFixed(2)}</text>
          <text x={c} y={size*0.81} textAnchor="middle">{qVals[2].toFixed(2)}</text>
          <text x={size*0.26} y={c+3} textAnchor="middle">{qVals[3].toFixed(2)}</text>
        </g>
        {!allEq && !isAgent && <text x={c} y={c+7} textAnchor="middle" fontSize="22" fill={greedyColor} fontWeight="bold" pointerEvents="none">{ACTION_LABELS[bestIdx]}</text>}
        {showOptimal && disagrees && (
          <g pointerEvents="none">
            <circle cx={6} cy={6} r="8" fill={accentGood()} opacity="0.85" />
            <text x={6} y={9} textAnchor="middle" fontSize="11" fill="white" fontWeight="bold">{ACTION_LABELS[optimalAction]}</text>
          </g>
        )}
        {cellType === 4 && !isAgent && (
          <g pointerEvents="none">
            <circle cx={size-11} cy={size-11} r="9" fill="#3b82f6" />
            <text x={size-11} y={size-7} textAnchor="middle" fontSize="11" fontWeight="bold" fill="white">S</text>
          </g>
        )}
        {probeColor && <g pointerEvents="none"><circle cx={size-7} cy={7} r="5" fill={probeColor} stroke="white" strokeWidth="1.5" /></g>}
        {isAgent && (
          <g pointerEvents="none">
            <circle cx={c} cy={c} r="14" fill={lastAction?.explore ? '#f59e0b' : '#3b82f6'} stroke="white" strokeWidth="2.5" />
            {lastAction && <text x={c} y={c+5} textAnchor="middle" fontSize="14" fill="white" fontWeight="bold">{ACTION_LABELS[lastAction.a]}</text>}
          </g>
        )}
        <rect x={sw/2} y={sw/2} width={size-sw} height={size-sw} fill="none" stroke={stroke} strokeWidth={sw} />
      </g>
    );
  }

  let fill = 'hsl(40, 15%, 96%)', bigText = null, bigTextColor = greedyColor, cornerText = null;
  if (viewMode === 'vstar') { fill = qColor(vStar, vStarScale); bigText = optimalAction != null ? ACTION_LABELS[optimalAction] : null; bigTextColor = accentGood(); cornerText = vStar.toFixed(2); }
  else if (viewMode === 'tgtdiff') { const diff = Math.abs(mq - tgtMaxQ); fill = diffColor(diff, tgtDiffScale); bigText = ACTION_LABELS[bestIdx]; bigTextColor = '#581c87'; cornerText = diff.toFixed(3); }
  else if (viewMode === 'verror') { fill = errorColor(errorVal, errorScale); bigText = ACTION_LABELS[bestIdx]; bigTextColor = errorVal > errorScale * 0.5 ? accentBad() : accentGood(); cornerText = errorVal.toFixed(3); }
  else if (viewMode === 'visits') { fill = visitColor(visitCount, maxVisits); bigText = !allEq ? ACTION_LABELS[bestIdx] : null; bigTextColor = '#0f172a'; cornerText = `${visitCount}`; }
  else if (viewMode === 'vprop') {
    // Diverging fill on the signed value change since last episode: rising value
    // (credit arriving) in blue, falling in orange. Magnitude scaled to the max |ΔV|.
    const d = vDeltaVal || 0;
    const t = clamp(Math.abs(d) / (vDeltaScale || 1e-6), 0, 1);
    if (t < 0.02) fill = 'hsl(220, 12%, 95%)';
    else fill = d > 0 ? `hsl(205, 80%, ${92 - t * 48}%)` : `hsl(28, 85%, ${92 - t * 40}%)`;
    bigText = Math.abs(d) > (vDeltaScale || 1) * 0.15 ? (d > 0 ? '↑' : '↓') : null;
    bigTextColor = d > 0 ? '#0c4a6e' : '#7c2d12';
    cornerText = Math.abs(d) > 1e-3 ? (d > 0 ? '+' : '') + d.toFixed(2) : null;
  }
  else if (viewMode === 'trajectories') {
    // Light neutral backdrop: walls (#475569) must stay the DARK elements as in
    // every other view. A dark floor here made walls look lighter than open
    // space, inverting the grid's visual language. The path colours (emerald /
    // red / amber) read clearly against a light ground.
    fill = 'hsl(220, 16%, 88%)';
    bigText = !allEq ? ACTION_LABELS[bestIdx] : null;
    bigTextColor = '#64748b';
    cornerText = null;
  }

  return (
    <g onClick={onClick} style={{ cursor: cur }}>
      <rect width={size} height={size} fill={fill} />
      {cornerText && <text x={c} y={size*0.32} textAnchor="middle" fontSize="11" fill="#1f2937" fontFamily="ui-monospace, Menlo, monospace" pointerEvents="none">{cornerText}</text>}
      {bigText && !isAgent && <text x={c} y={size*0.78} textAnchor="middle" fontSize="22" fill={bigTextColor} fontWeight="bold" pointerEvents="none">{bigText}</text>}
      {cellType === 4 && !isAgent && (
        <g pointerEvents="none">
          <circle cx={size-11} cy={size-11} r="9" fill="#3b82f6" />
          <text x={size-11} y={size-7} textAnchor="middle" fontSize="11" fontWeight="bold" fill="white">S</text>
        </g>
      )}
      {probeColor && <g pointerEvents="none"><circle cx={size-7} cy={7} r="5" fill={probeColor} stroke="white" strokeWidth="1.5" /></g>}
      {isAgent && (
        <g pointerEvents="none">
          <circle cx={c} cy={c} r="14" fill={lastAction?.explore ? '#f59e0b' : '#3b82f6'} stroke="white" strokeWidth="2.5" />
          {lastAction && <text x={c} y={c+5} textAnchor="middle" fontSize="14" fill="white" fontWeight="bold">{ACTION_LABELS[lastAction.a]}</text>}
        </g>
      )}
      <rect x={sw/2} y={sw/2} width={size-sw} height={size-sw} fill="none" stroke={stroke} strokeWidth={sw} />
    </g>
  );
}

// Build a screen-reader label for a grid cell: its type, position, and (for
// empty/start cells) the greedy action and its Q-value.
function cellAriaLabel(x, y, cellType, qMap, env) {
  const pos = `row ${y + 1}, column ${x + 1}`;
  if (cellType === 1) return `Wall, ${pos}`;
  if (cellType === 2) return `Goal, reward plus one, ${pos}`;
  if (cellType === 3) return `Hazard, reward minus one, ${pos}`;
  const isStart = cellType === 4;
  const q = qMap?.cells?.[y]?.[x]?.q;
  let detail = '';
  if (q && q.length === 4) {
    let best = 0; for (let i = 1; i < 4; i++) if (q[i] > q[best]) best = i;
    const spread = Math.max(...q) - Math.min(...q);
    detail = spread < 1e-6 ? ', no preferred action yet' : `, best action ${ACTION_NAMES[best].toLowerCase()}, value ${q[best].toFixed(2)}`;
  }
  return `${isStart ? 'Start cell' : 'Empty cell'}, ${pos}${detail}`;
}

function GridPanel({ algorithm, grid, qMap, qScale, env, lastTrans, lastAction, editMode, setEditMode, probeMode, setProbeMode, probes,
  brush, setBrush, onCellClick, syncFlash, bgStatus, optimal, showOptimal, setShowOptimal, viewMode, setViewMode,
  tgtMap, vStarScale, tgtDiffScale, errorScale, maxVisits, visits, rollout, trajectories = [], vDelta }) {
  const CS = 72;
  const totalW = W * CS, totalH = Hg * CS;
  // Roving tabindex for the accessible grid overlay: one tab stop into the grid,
  // then arrow keys move focus between cells (the ARIA grid pattern the role and
  // label promise). Focus is only moved programmatically when it already sits
  // inside the grid, so we never steal it on mount or on re-render.
  const gridNavRef = useRef(null);
  const [focusCell, setFocusCell] = useState({ x: 0, y: 0 });
  useEffect(() => {
    const host = gridNavRef.current;
    if (!host || !host.contains(document.activeElement)) return;
    const el = host.querySelector(`[data-cell="${focusCell.x}-${focusCell.y}"]`);
    if (el) el.focus();
  }, [focusCell]);
  const setView = (m) => { setViewMode(m); if (m !== 'q') setEditMode(false); };
  // Scale for the value-propagation (#1) view: largest |ΔV| across cells.
  let vDeltaScale = 1e-6;
  if (vDelta) for (let i = 0; i < vDelta.length; i++) { const a = Math.abs(vDelta[i]); if (a > vDeltaScale) vDeltaScale = a; }
  const resultEmoji = rollout.result === 'goal' ? '🚩' : rollout.result === 'hazard' ? '💀' : '⌛';
  const resultColor = rollout.result === 'goal' ? 'border-emerald-500 bg-emerald-950/95 text-emerald-200'
                    : rollout.result === 'hazard' ? 'border-red-500 bg-red-950/95 text-red-200'
                    : 'border-amber-500 bg-amber-950/95 text-amber-200';

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">Grid World</h2>
        <div className="flex items-center gap-1.5 flex-wrap">
          <div data-tutor="view-modes"><ViewModePicker viewMode={viewMode} setViewMode={setView} algorithm={algorithm} /></div>
          <button onClick={() => setShowOptimal(!showOptimal)} disabled={viewMode !== 'q'}
            title={viewMode !== 'q' ? 'Switch to the Q-values view to use this' : 'Overlay the optimal action and flag cells where the agent disagrees'}
            className={`px-2.5 py-1 rounded border text-xs flex items-center gap-1.5 transition disabled:opacity-30 ${
              showOptimal ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}><Award size={12} /> Optimal</button>
          <button onClick={() => { setProbeMode(!probeMode); if (!probeMode) setEditMode(false); }} disabled={!!bgStatus}
            title={bgStatus ? 'Disabled while background training runs' : 'Click cells to track their max-Q value over time (up to 4)'}
            className={`px-2.5 py-1 rounded border text-xs flex items-center gap-1.5 transition disabled:opacity-30 ${
              probeMode ? 'bg-orange-600/20 border-orange-500 text-orange-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}><Crosshair size={12} /> Probes</button>
          <button onClick={() => { setEditMode(!editMode); if (!editMode) setProbeMode(false); }} disabled={!!bgStatus || viewMode !== 'q'}
            title={bgStatus ? 'Disabled while background training runs' : viewMode !== 'q' ? 'Switch to the Q-values view to edit the grid' : 'Paint walls, goals, hazards, and the start cell'}
            className={`px-2.5 py-1 rounded border text-xs flex items-center gap-1.5 transition disabled:opacity-30 ${
              editMode ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}><Edit3 size={12} /> Edit</button>
        </div>
      </div>

      <div className="flex flex-col gap-4 items-start">
        <div data-tutor="grid" style={{ maxWidth: totalW + 16 }} className="relative bg-slate-100 rounded-lg p-2 w-full">
          <svg viewBox={`0 0 ${totalW} ${totalH}`} width={totalW} height={totalH}
            role="img" aria-label={`${W} by ${Hg} grid world, ${viewMode} view`}
            style={{ width: '100%', maxWidth: totalW, height: 'auto', display: 'block' }}
            className={`rounded ${syncFlash > 0 ? 'ring-2 ring-amber-400' : ''} ${bgStatus ? 'opacity-60' : ''} ${rollout.active ? 'ring-2 ring-emerald-500' : ''}`}>
            {grid.map((row, y) => row.map((cellType, x) => {
              const qVals = qMap?.cells?.[y]?.[x]?.q ?? [0, 0, 0, 0];
              const isAgent = env && env.x === x && env.y === y;
              const isUpdSrc = !rollout.active && !!lastTrans && lastTrans.sx === x && lastTrans.sy === y;
              const isUpdDst = !rollout.active && !!lastTrans && lastTrans.dx === x && lastTrans.dy === y && !(lastTrans.sx === x && lastTrans.sy === y);
              const optimalAction = optimal?.policy?.[y * W + x] ?? -1;
              const vStar = optimal?.V?.[y * W + x] ?? 0;
              const tgtCell = tgtMap?.cells?.[y]?.[x];
              const tgtMaxQ = tgtCell ? Math.max(...tgtCell.q) : 0;
              const maxQ = qVals.length ? Math.max(...qVals) : 0;
              const errorVal = Math.abs(maxQ - vStar);
              const visitCount = visits ? visits[y * W + x] : 0;
              const vDeltaVal = vDelta ? vDelta[y * W + x] : 0;
              const probe = probes.find(p => p.x === x && p.y === y);
              return (
                <g key={`${x}-${y}`} transform={`translate(${x*CS}, ${y*CS})`}>
                  <Cell cellType={cellType} qVals={qVals} isAgent={isAgent} isUpdSrc={isUpdSrc} isUpdDst={isUpdDst}
                    qScale={qScale} lastAction={lastAction} editMode={editMode} probeMode={probeMode}
                    onClick={() => onCellClick(x, y)} size={CS}
                    optimalAction={optimalAction >= 0 ? optimalAction : null} showOptimal={showOptimal}
                    viewMode={viewMode} vStar={vStar} vStarScale={vStarScale}
                    tgtMaxQ={tgtMaxQ} tgtDiffScale={tgtDiffScale}
                    errorVal={errorVal} errorScale={errorScale} visitCount={visitCount} maxVisits={maxVisits}
                    vDeltaVal={vDeltaVal} vDeltaScale={vDeltaScale}
                    probeColor={probe?.color} />
                </g>
              );
            }))}
            {viewMode === 'trajectories' && trajectories.length > 0 && (
              <g pointerEvents="none">
                {trajectories.map((tr, ti) => {
                  // newest = strongest; fade older ones
                  const age = trajectories.length - 1 - ti; // 0 = newest
                  const opacity = Math.max(0.12, 0.85 - age * 0.12);
                  const col = tr.outcome === 'goal' ? '#10b981' : tr.outcome === 'hazard' ? '#ef4444' : '#f59e0b';
                  // small deterministic jitter per path so overlaps are visible
                  const jx = ((ti * 13) % 7 - 3) * 1.6;
                  const jy = ((ti * 7) % 7 - 3) * 1.6;
                  const pts = tr.path.map(p => `${p.x*CS + CS/2 + jx},${p.y*CS + CS/2 + jy}`).join(' ');
                  return (
                    <g key={ti}>
                      <polyline points={pts} fill="none" stroke={col} strokeWidth={age === 0 ? 3.5 : 2}
                        strokeOpacity={opacity} strokeLinecap="round" strokeLinejoin="round" />
                      {age === 0 && tr.path.length > 0 && (
                        <circle cx={tr.path[0].x*CS + CS/2 + jx} cy={tr.path[0].y*CS + CS/2 + jy} r="4" fill={col} stroke="#0f172a" strokeWidth="1" />
                      )}
                    </g>
                  );
                })}
              </g>
            )}
            {rollout.path.length > 1 && (
              <g pointerEvents="none">
                <polyline points={rollout.path.map(p => `${p.x*CS + CS/2},${p.y*CS + CS/2}`).join(' ')}
                  fill="none" stroke="#10b981" strokeWidth="4" strokeOpacity="0.55" strokeLinecap="round" strokeLinejoin="round" />
                {rollout.path.slice(0, -1).map((p, i) => <circle key={i} cx={p.x*CS + CS/2} cy={p.y*CS + CS/2} r="3.5" fill="#059669" />)}
              </g>
            )}
          </svg>

          {/* Accessibility overlay: a focusable grid of buttons over the SVG so the
              grid is reachable by keyboard and screen readers. Interactive only in
              edit/probe mode (otherwise cells aren't actionable and would be 36
              dead tab stops). Positioned with percentages so it tracks the
              responsive SVG. Padding offset matches the wrapper's p-2 (8px). */}
          {(editMode || probeMode) && !bgStatus && (
            <div role="grid" aria-label={editMode ? 'Editable grid — arrow keys move, Enter paints' : 'Grid — arrow keys move, Enter probes the cell'}
              ref={gridNavRef}
              onKeyDown={(e) => {
                let dx = 0, dy = 0;
                if (e.key === 'ArrowLeft') dx = -1;
                else if (e.key === 'ArrowRight') dx = 1;
                else if (e.key === 'ArrowUp') dy = -1;
                else if (e.key === 'ArrowDown') dy = 1;
                else if (e.key === 'Home') { e.preventDefault(); setFocusCell(c => ({ ...c, x: 0 })); return; }
                else if (e.key === 'End') { e.preventDefault(); setFocusCell(c => ({ ...c, x: W - 1 })); return; }
                else return;
                e.preventDefault();
                setFocusCell(c => ({ x: clamp(c.x + dx, 0, W - 1), y: clamp(c.y + dy, 0, Hg - 1) }));
              }}
              className="absolute" style={{ top: 8, left: 8, right: 8, bottom: 8 }}>
              {grid.map((row, y) => (
                <div key={y} role="row" className="flex" style={{ height: `${100 / Hg}%` }}>
                  {row.map((cellType, x) => {
                    const label = cellAriaLabel(x, y, cellType, qMap, env);
                    const isFocus = focusCell.x === x && focusCell.y === y;
                    return (
                      <button key={x} role="gridcell" aria-label={label} title=""
                        data-cell={`${x}-${y}`}
                        tabIndex={isFocus ? 0 : -1}
                        onFocus={() => setFocusCell(c => (c.x === x && c.y === y ? c : { x, y }))}
                        onClick={() => onCellClick(x, y)}
                        className="focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset rounded-sm"
                        style={{ width: `${100 / W}%`, height: '100%', background: 'transparent', border: 'none', cursor: 'pointer' }} />
                    );
                  })}
                </div>
              ))}
            </div>
          )}
          {bgStatus && (
            <div className="absolute inset-0 flex items-center justify-center rounded-lg pointer-events-none">
              <div className="bg-slate-900/95 border border-purple-500/50 rounded-lg px-4 py-3 shadow-2xl flex items-center gap-3">
                <Loader2 size={22} className="animate-spin text-purple-400" />
                <div className="text-sm">
                  <div className="font-medium text-purple-200">Training <span className="font-mono">{bgStatus.target.toLocaleString()}</span> {bgStatus.mode}<TrainingDots /></div>
                  <div className="text-[11px] text-slate-400">UI will update when complete</div>
                </div>
              </div>
            </div>
          )}
          {rollout.active && !bgStatus && (
            <div className="absolute top-3 left-3 bg-emerald-500 text-slate-950 text-xs px-2 py-1 rounded font-bold shadow flex items-center gap-1">
              <Route size={12} /> Rolling out step {rollout.steps}
            </div>
          )}
          {rollout.result && !rollout.active && !bgStatus && (
            <div className={`absolute inset-x-3 top-3 border rounded-lg px-3 py-2 shadow-2xl text-sm font-medium flex items-center justify-center gap-2 ${resultColor}`}>
              <span>{resultEmoji}</span>
              <span>Rollout {rollout.result === 'goal' ? 'reached goal' : rollout.result === 'hazard' ? 'hit hazard' : 'timed out'} in {rollout.steps} steps · reward {rollout.totalReward >= 0 ? '+' : ''}{rollout.totalReward.toFixed(2)}</span>
            </div>
          )}
          {syncFlash > 0 && !bgStatus && !rollout.active && !rollout.result && (
            <div className="absolute top-3 right-3 bg-amber-500 text-slate-950 text-xs px-2 py-1 rounded font-bold animate-pulse shadow">TARGET SYNC</div>
          )}
        </div>

        <div className="flex-1 text-sm space-y-3 min-w-0">
          {editMode && (
            <div className="space-y-2">
              <div className="text-xs text-slate-400">Click a cell to paint:</div>
              <div className="flex gap-1.5 flex-wrap">
                <BrushBtn cur={brush} val={0} setBrush={setBrush} color="bg-slate-600">Empty</BrushBtn>
                <BrushBtn cur={brush} val={1} setBrush={setBrush} color="bg-slate-500">Wall</BrushBtn>
                <BrushBtn cur={brush} val={2} setBrush={setBrush} color="bg-amber-700">Goal 🚩</BrushBtn>
                <BrushBtn cur={brush} val={3} setBrush={setBrush} color="bg-red-800">Hazard 💀</BrushBtn>
                <BrushBtn cur={brush} val={4} setBrush={setBrush} color="bg-blue-700">Start</BrushBtn>
              </div>
            </div>
          )}
          {probeMode && (
            <div className="text-xs text-slate-400 bg-orange-950/30 border border-orange-700/40 rounded p-2">
              <span className="font-medium text-orange-300">Probe mode:</span> click cells to track max-Q (up to 4).
              {probes.length > 0 && <span className="block mt-1">{probes.length}/4 selected</span>}
            </div>
          )}
          <Legend showOptimal={showOptimal} viewMode={viewMode} probesActive={probes.length > 0} />
          <div className="text-xs text-slate-400 leading-relaxed">
            <ViewModeExplain viewMode={viewMode} showOptimal={showOptimal} algorithm={algorithm} />
          </div>
        </div>
      </div>
    </div>
  );
}

function ViewModePicker({ viewMode, setViewMode, algorithm }) {
  const opts = [
    { id: 'q', label: 'Q', icon: <Grid3x3 size={11} />, tip: "Learned Q-values (default)." },
    { id: 'vstar', label: 'V*', icon: <Award size={11} />, tip: 'Optimal V*(s) — ground truth.' },
    { id: 'verror', label: 'Err', icon: <AlertTriangle size={11} />, tip: '|max-Q − V*|. Green=accurate, red=off.' },
    { id: 'vprop', label: 'Flow', icon: <Activity size={11} />, tip: 'Value propagation: how max-Q changed since last episode. See credit flow back from the goal.' },
    { id: 'trajectories', label: 'Paths', icon: <Route size={11} />, tip: 'Recent episode trajectories, fading with age, colored by outcome.' },
    { id: 'visits', label: 'Visits', icon: <Eye size={11} />, tip: 'How often each cell has been visited.' },
  ];
  if (algorithm === 'dqn') opts.push({ id: 'tgtdiff', label: 'Δtgt', icon: <Activity size={11} />, tip: '|Q_online − Q_target|; resets on sync.' });
  return (
    <div className="flex bg-slate-800 rounded border border-slate-700 p-0.5 text-xs">
      {opts.map(o => (
        <button key={o.id} onClick={() => setViewMode(o.id)} title={o.tip}
          className={`px-2 py-0.5 rounded transition flex items-center gap-1 ${viewMode === o.id ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}>
          {o.icon}{o.label}
        </button>
      ))}
    </div>
  );
}

function ViewModeExplain({ viewMode, showOptimal, algorithm }) {
  const cb = _palette === 'cbsafe';
  const matchWord = cb ? 'teal' : 'green';
  const wrongWord = cb ? 'orange' : 'red';
  const lowWord = cb ? 'light' : 'green';
  const highWord = cb ? 'deep orange' : 'red';
  if (viewMode === 'q') {
    const what = algorithm === 'tabular' ? 'Q-table values' : 'network-predicted Q-values';
    return (<>
      <div className="font-medium text-slate-300 mb-1">Q-values view ({what})</div>
      Four wedges per cell (top=↑, right=→, bottom=↓, left=←) colored by Q-value. Center letter is the greedy action.
      {showOptimal && <> With <span className="text-emerald-400">Optimal</span> on: {matchWord} = matches, {wrongWord} = disagrees.</>}
    </>);
  }
  if (viewMode === 'vstar') return (<><div className="font-medium text-slate-300 mb-1">Optimal V* view</div>
    Cell color = optimal value V*(s). The arrow is the optimal action. Ground truth.</>);
  if (viewMode === 'verror') return (<><div className="font-medium text-slate-300 mb-1">Value error view</div>
    Cell color = |max-Q(s) − V*(s)|. {lowWord.charAt(0).toUpperCase() + lowWord.slice(1)} = matches truth, {highWord} = still way off. Watch error shrink from goal outward.</>);
  if (viewMode === 'visits') return (<><div className="font-medium text-slate-300 mb-1">State visitation view</div>
    Cell color = how many times the agent has been here (log scale). White = unexplored.</>);
  if (viewMode === 'vprop') return (<><div className="font-medium text-slate-300 mb-1">Value propagation (flow)</div>
    Cell color = how much max-Q <em>changed</em> since the last episode. Blue ↑ = value rising (credit arriving), orange ↓ = falling. Early on you'll see a wavefront spreading back from the goal — that's credit assignment in action.</>);
  if (viewMode === 'trajectories') return (<><div className="font-medium text-slate-300 mb-1">Trajectory overlay (paths)</div>
    Recent episode paths, newest brightest, fading with age. Green = reached goal, red = hit hazard, amber = timed out. Watch routes converge as the policy sharpens.</>);
  return (<><div className="font-medium text-slate-300 mb-1">Online–target Q diff view</div>
    Cell color = |max Q<sub>online</sub>(s) − max Q<sub>target</sub>(s)|. Collapses to ~0 on each sync.</>);
}

function TrainingDots() {
  return (
    <span className="inline-block w-5 text-left">
      <style>{`@keyframes dqn-dots { 0%, 20% { opacity: 0; } 50% { opacity: 1; } 100% { opacity: 0; } }
        .dqn-d1 { animation: dqn-dots 1.4s infinite 0s; }
        .dqn-d2 { animation: dqn-dots 1.4s infinite 0.2s; }
        .dqn-d3 { animation: dqn-dots 1.4s infinite 0.4s; }`}</style>
      <span className="dqn-d1">.</span><span className="dqn-d2">.</span><span className="dqn-d3">.</span>
    </span>
  );
}

function BrushBtn({ cur, val, setBrush, color, children }) {
  return (
    <button onClick={() => setBrush(val)}
      className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 ${cur === val ? 'ring-2 ring-blue-400' : ''} ${color} text-white`}>
      {children}
    </button>
  );
}

function Legend({ showOptimal, viewMode, probesActive }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
      <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-blue-500" />Greedy</div>
      <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-amber-500" />Exploring</div>
      <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-amber-700" />Goal +1</div>
      <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-800" />Hazard −1</div>
      <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-slate-500" />Wall</div>
      <div className="flex items-center gap-1.5"><Route size={11} className="text-emerald-500" />Rollout path</div>
      {viewMode === 'q' && showOptimal && <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm border-2" style={{ borderColor: accentBad() }} />Wrong action</div>}
      {probesActive && <div className="flex items-center gap-1.5"><Crosshair size={11} className="text-orange-400" />Probe</div>}
    </div>
  );
}

// Lightweight confirm modal for destructive actions (wiping a trained agent).
// Esc / backdrop-click cancels; Enter confirms. Accent defaults to rose since
// these are always destructive in this app.
function ConfirmDialog({ title, body, confirmLabel = 'Continue', cancelLabel = 'Cancel', onConfirm, onCancel }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
      else if (e.key === 'Enter') { e.preventDefault(); onConfirm(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onConfirm, onCancel]);
  return (
    <>
      <style>{`
        @keyframes cd-overlay-in { from { opacity: 0; } to { opacity: 1; } }
        @keyframes cd-card-in { from { opacity: 0; transform: translateY(8px) scale(0.98); } to { opacity: 1; transform: none; } }
        .cd-overlay { animation: cd-overlay-in 0.15s ease-out both; }
        .cd-card { animation: cd-card-in 0.2s cubic-bezier(0.2,0.8,0.2,1) both; }
      `}</style>
      <div className="cd-overlay fixed inset-0 z-[170] bg-slate-950/75 backdrop-blur-sm" onClick={onCancel} />
      <div className="fixed inset-0 z-[180] flex items-center justify-center p-4 pointer-events-none">
        <div className="cd-card pointer-events-auto w-full max-w-sm bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-5"
          onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
          <div className="flex items-start gap-3 mb-3">
            <div className="w-9 h-9 rounded-lg bg-rose-500/15 border border-rose-500/40 flex items-center justify-center shrink-0">
              <AlertTriangle size={18} className="text-rose-300" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100 leading-tight mb-1">{title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{body}</p>
            </div>
          </div>
          <div className="flex items-center justify-end gap-2 mt-4">
            <button onClick={onCancel} autoFocus
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200">
              {cancelLabel}
            </button>
            <button onClick={onConfirm}
              className="px-3 py-1.5 text-xs rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-medium flex items-center gap-1.5">
              <RotateCcw size={12} /> {confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

// ===== Save / load sessions =====
function timeAgo(ts) {
  if (!ts) return '';
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return 'just now';
  const m = Math.floor(s / 60); if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60); if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24); return `${d}d ago`;
}

function SaveLoadPanel({ saves, storageReady, saveStatus, onSave, onLoad, onDelete, onExport, disabled, flat }) {
  const [name, setName] = useState('');
  const [confirmDel, setConfirmDel] = useState(null); // key pending delete confirm
  return (
    <div className={cardCls(flat)}>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          Sessions
          <Info text="Save the current agent (weights or Q-table), grid, hyperparameters, and training history to this browser. Reload anytime — even after closing the tab." w={300} />
        </h2>
        <div className="flex items-center gap-2">
          {saveStatus && (
            <span className={`text-xs px-2 py-0.5 rounded ${saveStatus.kind === 'ok' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
              {saveStatus.text}
            </span>
          )}
          <button onClick={onExport}
            title="Download per-episode training data as a CSV file"
            className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs flex items-center gap-1.5">
            <Download size={12} /> CSV
          </button>
        </div>
      </div>

      {!storageReady ? (
        <div className="text-xs text-slate-500 py-3">
          Persistent storage isn&apos;t available in this environment, so sessions can&apos;t be saved here.
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-3">
            <input
              type="text" value={name} onChange={e => setName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !disabled) { onSave(name); setName(''); } }}
              placeholder="Name this save (optional)"
              disabled={disabled}
              className="flex-1 min-w-0 px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-100 text-sm placeholder:text-slate-600 focus:outline-none focus:border-emerald-500 disabled:opacity-40" />
            <button onClick={() => { onSave(name); setName(''); }} disabled={disabled}
              title={disabled ? 'Wait for training to finish' : 'Save the current session'}
              className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm flex items-center gap-1.5 shrink-0 disabled:opacity-40 disabled:cursor-not-allowed">
              <Save size={14} /> Save
            </button>
          </div>

          {saves.length === 0 ? (
            <div className="text-xs text-slate-500 py-2">No saved sessions yet.</div>
          ) : (
            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
              {saves.map((s) => (
                <div key={s.key} className="flex items-center gap-2 bg-slate-800/50 rounded-lg px-2.5 py-1.5 border border-slate-800">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-slate-200 truncate">{s.name}</div>
                    <div className="text-[10px] text-slate-500 flex items-center gap-2 flex-wrap">
                      <span className={`px-1 rounded ${s.algorithm === 'tabular' ? 'text-emerald-300' : 'text-sky-300'}`}>
                        {s.algorithm === 'tabular' ? 'Tabular' : 'DQN'}
                      </span>
                      <span className="font-mono">{(s.totalSteps || 0).toLocaleString()} steps</span>
                      <span>{timeAgo(s.savedAt)}</span>
                    </div>
                  </div>
                  <button onClick={() => onLoad(s.key)} disabled={disabled}
                    title="Load this session"
                    className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-100 text-xs flex items-center gap-1 shrink-0 disabled:opacity-40 disabled:cursor-not-allowed">
                    <FolderOpen size={12} /> Load
                  </button>
                  {confirmDel === s.key ? (
                    <button onClick={() => { onDelete(s.key, s.name); setConfirmDel(null); }}
                      title="Confirm delete"
                      className="px-2 py-1 rounded bg-rose-600 hover:bg-rose-500 text-white text-xs shrink-0">
                      Sure?
                    </button>
                  ) : (
                    <button onClick={() => setConfirmDel(s.key)}
                      title="Delete this session"
                      className="p-1 rounded text-slate-500 hover:text-rose-300 hover:bg-rose-500/10 shrink-0">
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Kbd({ children }) {
  return (
    <kbd className="px-1.5 py-0.5 rounded border border-slate-600 bg-slate-800 text-slate-300 font-mono text-[10px] leading-none">
      {children}
    </kbd>
  );
}

// Logarithmic speed slider: the slider's internal position is in [0, 1000] and
// maps log-style to a real speed in [1, 500] steps/sec. Linear was unusable —
// 1–30/s (the perceptually meaningful range) occupied only 6% of the bar.
// Below the slider are four clickable preset labels (Slow / Normal / Fast / Max)
// for one-tap jumps.
const SPEED_MIN = 1;
const SPEED_MAX = 500;
const SPEED_LOG_MAX = Math.log(SPEED_MAX);
const speedToPos = (s) => Math.round(1000 * Math.log(Math.max(SPEED_MIN, s)) / SPEED_LOG_MAX);
const posToSpeed = (p) => Math.max(SPEED_MIN, Math.min(SPEED_MAX, Math.round(Math.exp(p / 1000 * SPEED_LOG_MAX))));
const SPEED_STOPS = [
  { label: 'Slow',   speed: 3 },
  { label: 'Normal', speed: 20 },
  { label: 'Fast',   speed: 100 },
  { label: 'Max',    speed: 500 },
];

function SpeedSlider({ speed, setSpeed, effectiveSpeed, narrationMode }) {
  const pos = speedToPos(speed);
  // A stop is "active" when speed is at or very near it (within 10% on the log
  // axis) — gives the click target a satisfying highlight without snapping.
  const activeStopIdx = (() => {
    let best = -1, bestD = Infinity;
    SPEED_STOPS.forEach((st, i) => {
      const d = Math.abs(speedToPos(st.speed) - pos);
      if (d < bestD) { bestD = d; best = i; }
    });
    return bestD <= 25 ? best : -1;
  })();
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-slate-400 text-xs">Speed</span>
        <input type="range" min="0" max="1000" value={pos}
          onChange={e => setSpeed(posToSpeed(+e.target.value))}
          className="w-32 accent-blue-500" aria-label="Simulation speed (steps per second, log scale)" />
        <span className="text-slate-300 w-16 tabular-nums text-xs">
          {narrationMode ? `${effectiveSpeed}/s` : `${speed}/s`}
          {narrationMode && speed > effectiveSpeed && (
            <span className="text-amber-400" title={`Narration caps speed at ${effectiveSpeed}/s for readability`}> ⓘ</span>
          )}
        </span>
      </div>
      <div className="flex items-center gap-1 pl-9">
        {SPEED_STOPS.map((st, i) => (
          <button key={st.label} onClick={() => setSpeed(st.speed)}
            title={`${st.speed}/s`}
            className={`px-1 text-[10px] rounded transition ${
              i === activeStopIdx ? 'text-blue-300 font-semibold' : 'text-slate-500 hover:text-slate-300'
            }`}>
            {st.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// Environment stochasticity controls. Slip and reward-noise are properties of
// the world, not of either agent, so they live in their own panel and affect A
// and B identically. Changing slip recomputes the optimal V* ground truth.
function EnvironmentPanel({ slipProb, setSlipProb, rewardNoise, setRewardNoise, disabled, flat }) {
  const stochastic = slipProb > 0 || rewardNoise > 0;
  return (
    <div className={cardCls(flat)}>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          Environment
          <Info text="Make the world stochastic. Slip randomizes the agent's action some of the time; reward noise jitters each step's reward. Stochasticity is exactly where vanilla DQN's overestimation bias shows up — and where Double DQN helps. Optimal V* is recomputed to account for slip." w={320} />
        </h2>
        <span className={`text-[10px] px-2 py-0.5 rounded ${stochastic ? 'bg-sky-500/15 text-sky-300' : 'bg-slate-800 text-slate-500'}`}>
          {stochastic ? 'Stochastic' : 'Deterministic'}
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-3 text-xs">
        <Slider label="Slip probability" val={slipProb} min={0} max={0.5} step={0.05} onChange={v => setSlipProb(v)} fmt={v => `${(v * 100).toFixed(0)}%`} disabled={disabled} />
        <Slider label="Reward noise (σ)" val={rewardNoise} min={0} max={0.5} step={0.05} onChange={v => setRewardNoise(v)} fmt={v => v.toFixed(2)} disabled={disabled} />
      </div>
      {stochastic && (
        <div className="mt-2.5 flex items-start gap-1.5 text-[11px] text-sky-300/90">
          <Sparkles size={12} className="mt-0.5 shrink-0" />
          <span>Try vanilla vs Double DQN in Compare mode now — the gap in the reward and overestimation curves should widen.</span>
        </div>
      )}
    </div>
  );
}

function ControlPanel({ running, setRunning, speed, setSpeed, effectiveSpeed, onReset, onStepOnce, onStepEpisode,
  onFastSteps, onFastEpisodes, onRollout, rolloutActive, narrationMode, setNarrationMode, bgStatus, presets, updateGrid }) {
  const [epCount, setEpCount] = useState(50);
  const isBg = !!bgStatus;
  const disabled = isBg || rolloutActive;
  // When a control is disabled, say why on hover. Priority: background training,
  // then an active rollout, then the animation loop.
  const busyReason = isBg ? 'Disabled while background training runs'
    : rolloutActive ? 'Disabled during rollout (no learning happens then)'
    : null;
  const whileRunning = running ? 'Pause the animation first' : null;
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-500 mr-1">Manual:</span>
        <button data-tutor="step-btn" onClick={onStepOnce} disabled={running || disabled}
          title={whileRunning || busyReason || 'Advance one transition (S)'}
          className="px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-white flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed">
          <SkipForward size={14} /> Step
        </button>
        <button onClick={onStepEpisode} disabled={running || disabled}
          title={whileRunning || busyReason || 'Run to end of current episode (E)'}
          className="px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-white flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed">
          <ChevronsRight size={14} /> Episode
        </button>
        <button onClick={() => setRunning(r => !r)} disabled={disabled}
          title={busyReason || (running ? 'Pause animation (Space)' : 'Animate continuously with learning (Space)')}
          className={`px-3 py-1.5 rounded-lg font-medium flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed ${
            running ? 'bg-amber-600 hover:bg-amber-500' : 'bg-blue-600 hover:bg-blue-500'
          } text-white`}>
          {running ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Animate</>}
        </button>
        <button data-tutor="rollout-btn" onClick={onRollout} disabled={running || disabled}
          className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
          title={whileRunning || busyReason || 'Run current greedy policy from start, with no learning (R)'}>
          <Route size={14} /> Rollout
        </button>
        <SpeedSlider speed={speed} setSpeed={setSpeed} effectiveSpeed={effectiveSpeed} narrationMode={narrationMode} />
        <button onClick={onReset} disabled={disabled}
          title={busyReason || 'Reset the agent and clear all learning'}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed ml-auto">
          <RotateCcw size={14} /> Reset
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
        <span className="text-xs text-slate-500 mr-1">Fast train:</span>
        <div data-tutor="fast-train" className="flex flex-wrap items-center gap-2">
          <button onClick={onFastSteps} disabled={running || disabled}
            title={whileRunning || busyReason || 'Train 5,000 steps in the background, then update the view'}
            className="px-3 py-1.5 rounded-lg bg-purple-600/80 hover:bg-purple-600 text-white flex items-center gap-1.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed">
            <FastForward size={14} /> +5k steps
          </button>
          <div className="flex items-center gap-1.5 bg-slate-800/50 rounded-lg pl-2 pr-1 py-0.5 border border-slate-700">
            <span className="text-xs text-slate-400">Episodes</span>
            <input type="number" value={epCount}
              onChange={e => { const v = parseInt(e.target.value, 10); setEpCount(isNaN(v) ? 1 : Math.max(1, Math.min(5000, v))); }}
              disabled={running || disabled} min="1" max="5000"
              title={whileRunning || busyReason || 'Number of episodes to fast-train'}
              className="w-16 px-1.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-100 text-sm font-mono disabled:opacity-40 focus:outline-none focus:border-blue-500" />
            <button onClick={() => onFastEpisodes(epCount)} disabled={running || disabled}
              title={whileRunning || busyReason || `Fast-train ${epCount} episodes in the background`}
              className="px-2.5 py-1 rounded bg-emerald-600/80 hover:bg-emerald-600 text-white flex items-center gap-1 text-sm disabled:opacity-40 disabled:cursor-not-allowed">
              <Zap size={12} /> Run
            </button>
          </div>
        </div>

        <button onClick={() => setNarrationMode(!narrationMode)} disabled={disabled}
          className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 border transition disabled:opacity-40 ${
            narrationMode ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
          }`} title={busyReason || 'Plain-English commentary every step (speed capped to 2/s)'}>
          <MessageSquare size={12} /> Narrate
        </button>

        <div data-tutor="presets" className="flex items-center gap-1 ml-auto flex-wrap">
          <span className="text-xs text-slate-400 mr-1">Preset:</span>
          {Object.entries(presets).map(([k, v]) => (
            <button key={k} onClick={() => updateGrid(v.grid)} disabled={disabled}
              title={busyReason || v.desc}
              className="px-2 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 capitalize disabled:opacity-40 disabled:cursor-not-allowed">
              {k}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-slate-500 pt-1">
        <span className="inline-flex items-center gap-1"><Kbd>Space</Kbd> {running ? 'pause' : 'animate'}</span>
        <span className="inline-flex items-center gap-1"><Kbd>S</Kbd> step</span>
        <span className="inline-flex items-center gap-1"><Kbd>E</Kbd> episode</span>
        <span className="inline-flex items-center gap-1"><Kbd>R</Kbd> rollout</span>
      </div>

      {narrationMode && (
        <div className="text-[11px] text-blue-300/80 bg-blue-950/30 border border-blue-800/40 rounded px-2 py-1.5 flex items-center gap-1.5">
          <MessageSquare size={11} /> Narration on — each step writes a full sentence below. Speed capped to 2/s for readability.
        </div>
      )}

      {isBg && (
        <div className="flex items-center gap-2 bg-purple-600/15 border border-purple-500/40 px-3 py-2 rounded-lg text-purple-200 text-sm">
          <Loader2 size={14} className="animate-spin" />
          <span>Training {bgStatus.target.toLocaleString()} {bgStatus.mode} in background</span>
          <TrainingDots />
          <span className="ml-auto text-xs text-purple-300/70">Page may feel sluggish — that's the agent training</span>
        </div>
      )}
    </div>
  );
}

// ============ Inspector: tabbed learning visuals beside the grid ============
// Replaces the old long scrolling right column + bottom technical section with a
// single panel whose tab bar switches between the different learning views:
// live status, the model internals (network or Q-table), training charts, the
// replay buffer, settings, and the activity log / save-load.
// Panels render inside the tabbed InspectorPanel (flat=true → no own card chrome,
// since the inspector provides it) or standalone (flat=false → full card).
const cardCls = (flat, pad = 'p-4', extra = '') =>
  (flat ? `${pad} ${extra}` : `bg-slate-900 rounded-xl border border-slate-800 ${pad} ${extra}`).trim();

function InspectorPanel(props) {
  const {
    tab, setTab, algorithm, isTech, isTabular, mode, compareOn, onStartTech, narrationMode,
    currentEp, agent, totalSteps, lastAction, lastLearn, policyMatch, phase, hp, setHp,
    probes, probeHist, optimalV,
    state, syncFlash, grid, qMap, optimalPolicy,
    epRewards, epRewardsB, matchHist, overestHist, losses, epsHist, doubleA, doubleB,
    buffer,
    slipProb, setSlipProb, rewardNoise, setRewardNoise, archChanged,
    eventLog, saves, storageReady, saveStatus, onSave, onLoad, onDelete, onExport, bgStatus,
  } = props;

  const tabs = [
    { id: 'status', label: 'Status', icon: <Gauge size={13} /> },
    { id: 'model', label: isTabular ? 'Q-table' : 'Network', icon: isTabular ? <Table2 size={13} /> : <Brain size={13} /> },
    { id: 'charts', label: 'Charts', icon: <Activity size={13} /> },
    ...(!isTabular ? [{ id: 'memory', label: 'Replay', icon: <Database size={13} /> }] : []),
    { id: 'settings', label: 'Settings', icon: <Cpu size={13} /> },
    { id: 'log', label: 'Log', icon: <MessageSquare size={13} /> },
  ];
  const activeExists = tabs.some(t => t.id === tab);
  const active = activeExists ? tab : 'status';

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 sticky top-4 flex flex-col" style={{ maxHeight: 'calc(100vh - 2rem)' }}>
      <div className="flex items-center gap-1 p-2 border-b border-slate-800 overflow-x-auto shrink-0">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 whitespace-nowrap transition ${
              active === t.id ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}>
            {t.icon}{t.label}
          </button>
        ))}
        <div className="ml-auto pr-1 shrink-0">
          <button onClick={onStartTech} title="Guided tour of these panels"
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-slate-800">
            <GraduationCap size={14} />
          </button>
        </div>
      </div>

      <div className="p-3 space-y-3 overflow-y-auto min-h-0">
        {active === 'status' && (
          <>
            <StatsPanel
              episode={currentEp.ep} step={currentEp.step} epReward={currentEp.reward}
              eps={agent?.eps ?? 1} totalSteps={totalSteps}
              lastAction={lastAction} lastLearn={lastLearn}
              policyMatch={policyMatch} isTech={isTech} phase={phase} flat />
            <BellmanPanel algorithm={algorithm} lastLearn={lastLearn} gamma={hp.gamma} alpha={hp.alpha} flat />
            {probes.length > 0 && <ProbePanel probes={probes} probeHist={probeHist.slice()} optimalV={optimalV} flat />}
          </>
        )}

        {active === 'model' && (
          isTabular
            ? <div data-tutor="qtable"><QTablePanel grid={grid} qMap={qMap} optimalV={optimalV} optimalPolicy={optimalPolicy} flat /></div>
            : <div data-tutor="network"><NetworkPanel agent={agent} state={state} lastAction={lastAction} syncFlash={syncFlash} flat /></div>
        )}

        {active === 'charts' && (
          <>
            <RewardChart data={epRewards.slice()} dataB={epRewardsB ? epRewardsB.slice() : null} />
            <div data-tutor="match-chart"><MatchHistChart data={matchHist.slice()} /></div>
            {!isTabular && <div data-tutor="overest-chart"><OverestChart data={overestHist.slice()} compareOn={compareOn} doubleA={doubleA} doubleB={doubleB} /></div>}
            <div data-tutor="loss-chart"><LossChart data={losses.slice()} algorithm={algorithm} /></div>
            <div data-tutor="eps-chart"><EpsChart data={epsHist.slice()} /></div>
          </>
        )}

        {active === 'memory' && !isTabular && (
          <div data-tutor="replay"><ReplayPanel buffer={buffer} flat /></div>
        )}

        {active === 'settings' && (
          <>
            <div data-tutor="hyperparams"><HpPanel algorithm={algorithm} hp={hp} setHp={setHp} archChanged={archChanged} flat /></div>
            <EnvironmentPanel slipProb={slipProb} setSlipProb={setSlipProb} rewardNoise={rewardNoise} setRewardNoise={setRewardNoise} disabled={!!bgStatus} flat />
          </>
        )}

        {active === 'log' && (
          <>
            <Commentary log={eventLog} mode={mode} narrationMode={narrationMode} flat />
            <SaveLoadPanel saves={saves} storageReady={storageReady} saveStatus={saveStatus}
              onSave={onSave} onLoad={onLoad} onDelete={onDelete} onExport={onExport} disabled={!!bgStatus} flat />
          </>
        )}
      </div>
    </div>
  );
}

function StatsPanel({ episode, step, epReward, eps, totalSteps, lastAction, lastLearn, policyMatch, isTech, phase, flat }) {
  const phaseColors = {
    slate: 'bg-slate-800/60 border-slate-700 text-slate-300',
    amber: 'bg-amber-500/10 border-amber-500/40 text-amber-200',
    sky: 'bg-sky-500/10 border-sky-500/40 text-sky-200',
    emerald: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-200',
  };
  const phaseDot = { slate: 'bg-slate-400', amber: 'bg-amber-400', sky: 'bg-sky-400', emerald: 'bg-emerald-400' };
  return (
    <div data-tutor="stats" className={cardCls(flat)}>
      <h2 className="font-semibold mb-3 flex items-center gap-2">Status</h2>
      {phase && (
        <div className={`mb-3 rounded-lg border px-3 py-2 ${phaseColors[phase.color] || phaseColors.slate}`}>
          <div className="flex items-center gap-2 mb-0.5">
            <span className={`inline-block w-2 h-2 rounded-full ${phaseDot[phase.color] || phaseDot.slate} ${phase.key !== 'idle' && phase.key !== 'converged' ? 'animate-pulse' : ''}`} />
            <span className="text-sm font-semibold">{phase.label}</span>
          </div>
          <p className="text-[11px] leading-snug opacity-90">{phase.desc}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="Episode" value={episode} />
        <Stat label="Step" value={step} />
        <Stat label="Episode reward" value={epReward.toFixed(2)} />
        <Stat label={<span>ε <Info text="Probability of taking a random action instead of the greedy one. Decays toward ε_min." /></span>} value={eps.toFixed(3)} />
        {isTech && <Stat label="Total train steps" value={totalSteps} />}
        {isTech && <Stat label="Last loss" value={lastLearn?.loss?.toFixed(4) ?? '—'} />}
      </div>
      {policyMatch && (
        <div className="mt-3 pt-3 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400 flex items-center gap-1">
                Policy match
                <Info text="Fraction of cells where the agent's greedy action matches optimal. 100% = converged." />
              </div>
              <div className="text-lg font-mono">
                <span className={policyMatch.pct === 1 ? 'text-emerald-400' : policyMatch.pct >= 0.7 ? 'text-amber-300' : 'text-red-400'}>
                  {policyMatch.match}/{policyMatch.total}
                </span>
                <span className="text-slate-400 text-sm ml-2">({(policyMatch.pct * 100).toFixed(0)}%)</span>
              </div>
            </div>
            {policyMatch.startV != null && (
              <div className="text-right">
                <div className="text-xs text-slate-400 flex items-center justify-end gap-1">
                  V*(start)
                  <Info text="Optimal value of the start state — your agent's max-Q at start should converge here." />
                </div>
                <div className="text-lg font-mono text-blue-300">{policyMatch.startV.toFixed(3)}</div>
              </div>
            )}
          </div>
          <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className={`h-full transition-all ${policyMatch.pct === 1 ? 'bg-emerald-500' : policyMatch.pct >= 0.7 ? 'bg-amber-500' : 'bg-red-500'}`}
              style={{ width: `${policyMatch.pct * 100}%` }} />
          </div>
        </div>
      )}
      {lastAction && (
        <div className="mt-3 pt-3 border-t border-slate-800 text-sm">
          <div className="text-xs text-slate-400">Last action</div>
          <div className="font-medium flex items-center gap-2">
            <span className="text-2xl">{ACTION_LABELS[lastAction.a]}</span>
            <span>{ACTION_NAMES[lastAction.a]}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${lastAction.explore ? 'bg-amber-500/20 text-amber-300' : 'bg-blue-500/20 text-blue-300'}`}>
              {lastAction.explore ? 'EXPLORE' : 'EXPLOIT'}
            </span>
          </div>
          {isTech && lastAction.q && (
            <div className="mt-2 grid grid-cols-4 gap-1 text-xs">
              {lastAction.q.map((q, i) => (
                <div key={i} className={`text-center p-1 rounded ${i === lastAction.a ? 'bg-blue-600/30' : 'bg-slate-800'}`}>
                  <div className="text-slate-400">Q[{ACTION_LABELS[i]}]</div>
                  <div className="font-mono">{q.toFixed(2)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (<div><div className="text-xs text-slate-400">{label}</div><div className="text-lg font-mono">{value}</div></div>);
}

function BellmanPanel({ algorithm, lastLearn, gamma, alpha, flat }) {
  const b = lastLearn?.bellman;
  const findOne = (s) => { if (!s) return [-1, -1]; for (let i = 0; i < s.length; i++) if (s[i] > 0.5) return [i % W, Math.floor(i / W)]; return [-1, -1]; };
  const isTabular = algorithm === 'tabular';
  return (
    <div data-tutor="bellman" className={cardCls(flat, 'p-3')}>
      <h3 className="text-sm font-medium mb-2 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          Bellman update
          <Info text={isTabular ? 'Tabular Q-learning update: nudge Q(s,a) by α·TD error.' : 'DQN update: sample from replay buffer, compute target via target network, Adam step on Huber loss.'} w={300} />
        </span>
        <span className="text-[10px] text-slate-500">{isTabular ? 'most recent transition' : 'first sample of batch'}</span>
      </h3>
      {!b && <div className="text-xs text-slate-500 italic py-2">{isTabular ? 'Take a step to see the update.' : 'Waiting for first training step (buffer needs ≥ batch_size transitions).'}</div>}
      {b && (() => {
        const [sx, sy] = findOne(b.state);
        const [ex, ey] = findOne(b.nextState);
        const dq = b.newQ - b.oldQ;
        return (
          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              <div><span className="text-slate-500">s = </span><span className="font-mono text-slate-200">({sx},{sy})</span></div>
              <div><span className="text-slate-500">a = </span><span className="font-mono text-slate-200">{ACTION_LABELS[b.action]}</span></div>
              <div><span className="text-slate-500">r = </span><span className="font-mono text-slate-200">{b.reward.toFixed(3)}</span></div>
              <div className="flex items-center gap-1 flex-wrap">
                <span className="text-slate-500">s' = </span>
                <span className="font-mono text-slate-200">({ex},{ey})</span>
                {b.terminal && <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300">terminal</span>}
              </div>
            </div>
            <div className="bg-slate-950 rounded p-2.5 font-mono text-[11px] leading-relaxed overflow-x-auto border border-slate-800">
              <div className="text-slate-500 text-[10px] flex items-center gap-1 flex-wrap">
                {isTabular ? (
                  <>target = r + γ · max<sub>a'</sub> Q(s', a')</>
                ) : b.doubleDQN ? (
                  <>
                    target = r + γ · Q<sub>tgt</sub>(s', argmax<sub>a'</sub> Q<sub>online</sub>(s', a'))
                    <Info text="Double DQN: the ONLINE network picks the next action (argmax), the TARGET network supplies its value. Decoupling the two removes vanilla DQN's overestimation bias." w={300} />
                  </>
                ) : (
                  <>
                    target = r + γ · max<sub>a'</sub> Q<sub>tgt</sub>(s', a')
                    <Info text="Vanilla DQN: bootstrap from the target network (delayed copy) for stability — it both selects and evaluates the next action." w={280} />
                  </>
                )}
              </div>
              {!isTabular && b.doubleDQN && !b.terminal && (
                <div className="text-slate-500 text-[10px]">
                  online picks a' = <span className="text-blue-300">{ACTION_LABELS[b.maxNextA]}</span>; target evaluates it → {b.maxNextQ.toFixed(3)}
                </div>
              )}
              <div className="text-slate-200">
                {' = '}<span>{b.reward.toFixed(2)}</span>{' + '}
                <span className="text-blue-300">{gamma.toFixed(2)}</span>{' × '}
                <span>{b.terminal ? '0' : b.maxNextQ.toFixed(3)}</span>
                {b.terminal && <span className="text-slate-500"> (terminal)</span>}
                {' = '}<span className="text-emerald-300 font-semibold">{b.target.toFixed(3)}</span>
              </div>
              <div className="text-slate-500 text-[10px] mt-2 flex items-center gap-1">
                δ = target − Q(s,a)
                <Info text="TD error — distance between current estimate and target." w={220} />
              </div>
              <div className="text-slate-200">
                {' = '}<span className="text-emerald-300">{b.target.toFixed(3)}</span>{' − '}
                <span>{b.oldQ.toFixed(3)}</span>{' = '}
                <span className="text-amber-300 font-semibold">{b.tdError.toFixed(3)}</span>
              </div>
              <div className="text-slate-500 text-[10px] mt-2">
                {isTabular ? <>Update: Q(s,a) ← Q(s,a) + α · δ</> : <>After Adam step on Huber(δ):</>}
              </div>
              {isTabular ? (
                <div className="text-slate-200">
                  {' = '}<span>{b.oldQ.toFixed(3)}</span>{' + '}
                  <span className="text-blue-300">{alpha.toFixed(2)}</span>{' × '}
                  <span className="text-amber-300">{b.tdError.toFixed(3)}</span>
                  {' = '}<span className="text-emerald-300 font-semibold">{b.newQ.toFixed(3)}</span>
                  <span className={`ml-2 ${dq >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>({dq >= 0 ? '+' : ''}{dq.toFixed(4)})</span>
                </div>
              ) : (
                <div className="text-slate-200">
                  Q(s,a): <span>{b.oldQ.toFixed(3)}</span> → <span className="text-emerald-300 font-semibold">{b.newQ.toFixed(3)}</span>
                  <span className={`ml-2 ${dq >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>({dq >= 0 ? '+' : ''}{dq.toFixed(4)})</span>
                </div>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}

function ProbePanel({ probes, probeHist, optimalV, flat }) {
  const data = useMemo(() => probeHist.map(d => { const o = { step: d.step }; d.values.forEach((v, i) => { o[`v${i}`] = +v.toFixed(3); }); return o; }), [probeHist]);
  return (
    <div className={cardCls(flat, 'p-3')}>
      <h3 className="text-sm font-medium mb-2 flex items-center gap-1.5">
        Probe states
        <Info text="Max-Q for selected cells over training. Dashed lines show optimal V*(s)." w={260} />
      </h3>
      <div className="space-y-1 mb-2">
        {probes.map((p, i) => {
          const v = optimalV[p.y * W + p.x];
          const last = data.length ? data[data.length - 1][`v${i}`] : null;
          return (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: p.color }} />
              <span className="font-mono text-slate-300">({p.x},{p.y})</span>
              <span className="text-slate-500">·</span>
              <span className="text-slate-400">max-Q: <span className="font-mono text-slate-200">{last != null ? last.toFixed(3) : '—'}</span></span>
              <span className="text-slate-500">·</span>
              <span className="text-slate-400">V*: <span className="font-mono text-blue-300">{v.toFixed(3)}</span></span>
            </div>
          );
        })}
      </div>
      {data.length > 1 ? (
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -25 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
            <XAxis dataKey="step" stroke="#64748b" fontSize={10} />
            <YAxis stroke="#64748b" fontSize={10} />
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
            {probes.map((p, i) => {
              const v = optimalV[p.y * W + p.x];
              return (
                <React.Fragment key={i}>
                  <ReferenceLine y={+v.toFixed(3)} stroke={p.color} strokeDasharray="3 3" strokeOpacity={0.5} />
                  <Line type="monotone" dataKey={`v${i}`} stroke={p.color} dot={false} strokeWidth={1.8} isAnimationActive={false} />
                </React.Fragment>
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      ) : (<div className="text-xs text-slate-500 italic py-3 text-center">Train for a bit — probe values sampled every 25 steps.</div>)}
    </div>
  );
}

function ComparePanel({ algorithm, hpB, setHpB, onReset, policyMatch, policyMatchB, agentB, epRewardsB, grid, qMapA, agentBObj, env, optimalPolicy, tick, disabled }) {
  const setField = (k, v) => setHpB({ ...hpB, [k]: v });
  const isTabular = algorithm === 'tabular';
  // Greedy action per cell for A (from its qMap) and B (computed from agent B).
  // Read-only: for tabular B we read existing rows without creating them.
  const policyA = useMemo(() => {
    const out = new Int8Array(W * Hg).fill(-1);
    if (!qMapA?.cells) return out;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const c = grid[y][x]; if (c !== 0 && c !== 4) continue;
      const cell = qMapA.cells[y]?.[x]; if (!cell) continue;
      let b = 0; for (let i = 1; i < 4; i++) if (cell.q[i] > cell.q[b]) b = i;
      out[y * W + x] = b;
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qMapA, grid, tick]);
  const policyB = useMemo(() => {
    const out = new Int8Array(W * Hg).fill(-1);
    if (!agentBObj || !env) return out;
    const isTab = agentBObj instanceof TabularAgent;
    for (let y = 0; y < Hg; y++) for (let x = 0; x < W; x++) {
      const c = grid[y][x]; if (c !== 0 && c !== 4) continue;
      let q;
      if (isTab) { const row = agentBObj.qTable.get(`${x},${y}`); if (!row) { continue; } q = row; }
      else { q = agentBObj.q.forward(env.encodeAt(x, y)).q; }
      let b = 0; for (let i = 1; i < 4; i++) if (q[i] > q[b]) b = i;
      out[y * W + x] = b;
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentBObj, env, grid, tick]);
  return (
    <div className="mt-4 bg-fuchsia-950/30 rounded-xl border border-fuchsia-700/40 p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          A/B Comparison ({isTabular ? 'Tabular' : 'DQN'})
          <Info text="Train a second agent (B) alongside A on the same grid, in lockstep, so reward curves are directly comparable." w={300} />
        </h2>
        <button onClick={onReset} disabled={disabled}
          title={disabled ? 'Disabled while background training runs' : 'Rebuild agent B with its hyperparameters'}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1.5 text-sm disabled:opacity-40">
          <RotateCcw size={14} /> Reset B
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800">
          <div className="text-xs font-medium text-fuchsia-300 mb-2">Agent B hyperparameters</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {isTabular ? (
              <Slider label="α (learning rate)" val={hpB.alpha} min={0.01} max={1.0} step={0.01} onChange={v => setField('alpha', v)} fmt={v => v.toFixed(2)} />
            ) : (
              <Slider label="Learning rate" val={hpB.lr} min={0.0005} max={0.05} step={0.0005} onChange={v => setField('lr', v)} fmt={v => v.toFixed(4)} />
            )}
            <Slider label="γ (discount)" val={hpB.gamma} min={0.5} max={0.99} step={0.01} onChange={v => setField('gamma', v)} fmt={v => v.toFixed(2)} />
            <Slider label="ε decay" val={hpB.epsDecay} min={0.9} max={0.999} step={0.001} onChange={v => setField('epsDecay', v)} fmt={v => v.toFixed(3)} />
            <Slider label="ε min" val={hpB.epsMin} min={0.01} max={0.5} step={0.01} onChange={v => setField('epsMin', v)} fmt={v => v.toFixed(2)} />
            {!isTabular && (
              <>
                <Slider label="Target sync" val={hpB.targetUpdate} min={5} max={500} step={5} onChange={v => setField('targetUpdate', Math.round(v))} fmt={v => v} />
                <Slider label="Batch size" val={hpB.batchSize} min={4} max={128} step={4} onChange={v => setField('batchSize', Math.round(v))} fmt={v => v} />
              </>
            )}
          </div>
          {!isTabular && (
            <label className="mt-2 flex items-center justify-between gap-2 cursor-pointer">
              <span className="text-xs text-slate-300 font-medium">Double DQN</span>
              <button onClick={() => setField('doubleDQN', !hpB.doubleDQN)}
                role="switch" aria-checked={!!hpB.doubleDQN}
                className={`relative w-9 h-5 rounded-full transition shrink-0 ${hpB.doubleDQN ? 'bg-fuchsia-500' : 'bg-slate-600'}`}>
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${hpB.doubleDQN ? 'translate-x-4' : ''}`} />
              </button>
            </label>
          )}
          <div className="mt-2 text-[10px] text-slate-500">Click Reset B to apply.</div>
        </div>
        <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800 space-y-3">
          <div className="text-xs font-medium text-fuchsia-300">Side-by-side metrics</div>
          <div className="grid grid-cols-2 gap-3">
            <CompareMetric label="A · Policy match" pct={policyMatch?.pct} count={policyMatch ? `${policyMatch.match}/${policyMatch.total}` : '—'} color="#3b82f6" />
            <CompareMetric label="B · Policy match" pct={policyMatchB?.pct} count={policyMatchB ? `${policyMatchB.match}/${policyMatchB.total}` : '—'} color="#e879f9" />
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div><div className="text-slate-400">B · ε</div><div className="font-mono text-slate-200">{agentB?.eps?.toFixed(3) ?? '—'}</div></div>
            <div><div className="text-slate-400">B · episodes</div><div className="font-mono text-slate-200">{epRewardsB?.length ?? 0}</div></div>
            <div><div className="text-slate-400">B · last reward</div><div className="font-mono text-slate-200">{epRewardsB?.length ? epRewardsB[epRewardsB.length - 1].reward.toFixed(2) : '—'}</div></div>
          </div>
        </div>
      </div>

      {/* Dual policy grids: see WHERE the two agents' greedy policies differ from
          optimal, not just the aggregate percentage. Green = matches optimal,
          red = doesn't. This is the spatial complement to the match metric. */}
      <div className="mt-4">
        <div className="flex items-center gap-1.5 mb-2 text-xs font-medium text-fuchsia-300">
          Learned policies vs optimal
          <Info text="Each cell shows the agent's greedy action arrow, colored green where it matches the optimal policy (from value iteration) and red where it doesn't. Compare which agent has more green and where each one goes wrong." w={320} />
        </div>
        <div className="grid grid-cols-2 gap-4 max-w-xl">
          <PolicyMiniGrid title="Agent A" titleColor="#3b82f6" grid={grid} policy={policyA} optimalPolicy={optimalPolicy} pct={policyMatch?.pct} />
          <PolicyMiniGrid title="Agent B" titleColor="#e879f9" grid={grid} policy={policyB} optimalPolicy={optimalPolicy} pct={policyMatchB?.pct} />
        </div>
      </div>
    </div>
  );
}

// Compact read-only grid showing one agent's greedy policy, with each cell
// colored by whether its action matches the optimal policy.
function PolicyMiniGrid({ title, titleColor, grid, policy, optimalPolicy, pct }) {
  const CS = 30, totalW = W * CS, totalH = Hg * CS;
  return (
    <div>
      <div className="flex items-center justify-between mb-1 text-[11px]">
        <span style={{ color: titleColor }} className="font-medium">{title}</span>
        {pct != null && <span className="text-slate-400 font-mono">{(pct * 100).toFixed(0)}% optimal</span>}
      </div>
      <svg viewBox={`0 0 ${totalW} ${totalH}`} width={totalW} height={totalH}
        style={{ width: '100%', maxWidth: totalW, height: 'auto', display: 'block' }} className="rounded bg-slate-100">
        {grid.map((row, y) => row.map((c, x) => {
          const idx = y * W + x;
          let fill = '#e2e8f0', arrow = null, arrowColor = '#334155';
          if (c === 1) fill = '#475569';                       // wall
          else if (c === 2) { fill = '#bbf7d0'; arrow = '★'; arrowColor = '#15803d'; }   // goal
          else if (c === 3) { fill = '#fecaca'; arrow = '✕'; arrowColor = '#b91c1c'; }   // hazard
          else {
            const a = policy[idx];
            const opt = optimalPolicy ? optimalPolicy[idx] : -1;
            if (a < 0) { fill = '#f1f5f9'; }                    // unvisited / no policy yet
            else {
              const matches = opt < 0 || a === opt;
              fill = matches ? '#bbf7d0' : '#fed7aa';           // green match / orange mismatch
              arrow = ACTION_LABELS[a];
              arrowColor = matches ? '#15803d' : '#c2410c';
            }
          }
          return (
            <g key={idx} transform={`translate(${x * CS}, ${y * CS})`}>
              <rect width={CS - 1} height={CS - 1} fill={fill} stroke="#cbd5e1" strokeWidth="0.5" />
              {arrow && <text x={CS / 2} y={CS / 2 + 5} textAnchor="middle" fontSize="14" fontWeight="bold" fill={arrowColor}>{arrow}</text>}
            </g>
          );
        }))}
      </svg>
    </div>
  );
}

function CompareMetric({ label, pct, count, color }) {
  return (
    <div>
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-base font-mono"><span style={{ color }}>{count}</span>{pct != null && <span className="text-slate-400 text-xs ml-2">({(pct * 100).toFixed(0)}%)</span>}</div>
      <div className="mt-1 h-1 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full transition-all" style={{ width: `${(pct ?? 0) * 100}%`, background: color }} />
      </div>
    </div>
  );
}

function RewardChart({ data, dataB }) {
  const merged = useMemo(() => {
    if (!data.length && (!dataB || !dataB.length)) return [];
    const w = 10;
    const aMap = {};
    data.forEach((d, i) => {
      const slice = data.slice(Math.max(0, i - w), i + 1);
      const avg = slice.reduce((a, b) => a + b.reward, 0) / slice.length;
      aMap[d.ep] = { ep: d.ep, smoothA: +avg.toFixed(3) };
    });
    if (dataB) {
      dataB.forEach((d, i) => {
        const slice = dataB.slice(Math.max(0, i - w), i + 1);
        const avg = slice.reduce((a, b) => a + b.reward, 0) / slice.length;
        aMap[d.ep] = { ...(aMap[d.ep] || { ep: d.ep }), smoothB: +avg.toFixed(3) };
      });
    }
    return Object.values(aMap).sort((x, y) => x.ep - y.ep);
  }, [data, dataB]);

  // Data-driven Y bounds. A fixed domain clipped the curve whenever reward noise
  // was enabled (accumulated noise easily exceeds the deterministic range), so we
  // fit to the data and pad, while always keeping the [-1, 1] band visible for
  // context so small-variation runs don't look artificially dramatic.
  const yDomain = useMemo(() => {
    let lo = -1, hi = 1;
    for (const d of merged) {
      for (const k of ['smoothA', 'smoothB']) {
        const v = d[k];
        if (typeof v === 'number' && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
    }
    const pad = Math.max(0.2, (hi - lo) * 0.08);
    return [+(lo - pad).toFixed(2), +(hi + pad).toFixed(2)];
  }, [merged]);

  return (
    <ChartCard title="Reward per Episode" hint={dataB ? 'A vs B (smoothed)' : 'Higher = goal reached efficiently'}>
      <ResponsiveContainer width="100%" height={dataB ? 160 : 140}>
        <LineChart data={merged} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
          <XAxis dataKey="ep" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} domain={yDomain} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          {dataB && <RLegend wrapperStyle={{ fontSize: 10 }} />}
          <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="smoothA" name="Agent A" stroke="#3b82f6" dot={false} strokeWidth={2} isAnimationActive={false} />
          {dataB && <Line type="monotone" dataKey="smoothB" name="Agent B" stroke="#e879f9" dot={false} strokeWidth={2} isAnimationActive={false} />}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function MatchHistChart({ data }) {
  return (
    <ChartCard title="Policy match (% optimal)" hint="Across episodes">
      <ResponsiveContainer width="100%" height={100}>
        <LineChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
          <XAxis dataKey="ep" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          <ReferenceLine y={100} stroke="#10b981" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="pct" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// Q-value overestimation: agent's greedy max-Q at the start cell vs the true
// optimal value V*(start). The dashed green line is ground truth. A curve sitting
// ABOVE it is overestimating — the signature vanilla DQN bias that Double DQN
// suppresses. In compare mode both agents are shown.
function OverestChart({ data, compareOn, doubleA, doubleB }) {
  const vstar = data.length ? data[data.length - 1].vstar : 0;
  const labelA = `A · max-Q(start)${doubleA ? ' [Double]' : ''}`;
  const labelB = `B · max-Q(start)${doubleB ? ' [Double]' : ''}`;
  return (
    <ChartCard title="Q-value overestimation" hint="max-Q(start) vs V*(start)">
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
          <XAxis dataKey="ep" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          {compareOn && <RLegend wrapperStyle={{ fontSize: 10 }} />}
          <ReferenceLine y={vstar} stroke="#10b981" strokeDasharray="4 3"
            label={{ value: 'V*', position: 'right', fill: '#10b981', fontSize: 10 }} />
          <Line type="monotone" dataKey="estA" name={labelA} stroke="#3b82f6" dot={false} strokeWidth={2} isAnimationActive={false} />
          {compareOn && <Line type="monotone" dataKey="estB" name={labelB} stroke="#e879f9" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function LossChart({ data, algorithm }) {
  return (
    <ChartCard title={algorithm === 'tabular' ? 'Squared TD error' : 'TD Loss'} hint={algorithm === 'tabular' ? 'Per-step ½·δ²' : 'Huber on sampled batches'}>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
          <XAxis dataKey="step" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          <Line type="monotone" dataKey="loss" stroke="#f43f5e" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function EpsChart({ data }) {
  return (
    <ChartCard title="Epsilon Decay" hint="ε controls exploration; decays per episode">
      <ResponsiveContainer width="100%" height={100}>
        <LineChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 2" />
          <XAxis dataKey="ep" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }} />
          <Line type="monotone" dataKey="eps" stroke="#a855f7" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function ChartCard({ title, hint, children }) {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-3">
      <div className="flex justify-between items-baseline mb-1">
        <h3 className="text-sm font-medium">{title}</h3>
        {hint && <span className="text-[10px] text-slate-500">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function Commentary({ log, mode, narrationMode, flat }) {
  const maxItems = narrationMode ? 20 : (mode === 'beginner' ? 6 : 12);
  return (
    <div className={cardCls(flat, 'p-3')}>
      <h3 className="text-sm font-medium mb-2 flex items-center gap-1.5">
        {narrationMode ? 'Step-by-step narration' : "What's happening"}
      </h3>
      <div className={`space-y-1.5 text-xs overflow-y-auto ${narrationMode ? 'max-h-72' : 'max-h-32'}`}>
        {log.length === 0 && <div className="text-slate-500">Press Train or Step to start...</div>}
        {log.slice(0, maxItems).map((e, i) => {
          if (e.type === 'narrate') {
            return (
              <div key={i} className="bg-slate-950/60 border-l-2 border-blue-600 px-2 py-1.5 rounded-r leading-relaxed">
                <span className="text-slate-600 tabular-nums text-[10px] mr-2">t={e.t}</span>
                <span className="text-slate-300">{e.text}</span>
              </div>
            );
          }
          return (
            <div key={i} className={`flex gap-2 ${e.type === 'sync' ? 'text-amber-300' : e.type === 'episode' ? 'text-blue-300' : 'text-slate-400'}`}>
              <span className="text-slate-600 tabular-nums w-12 shrink-0">t={e.t}</span>
              <span>{e.text}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Per-layer gradient-norm readout: a sparkline of the total gradient L2 norm
// over recent updates, per-layer bars, and a vanishing/exploding health hint.
// Reads diagnostics the network records during trainBatch (read-only).
function GradientHealth({ net }) {
  const perLayer = net.lastGradNorms || [];
  const hist = net.gradNormHist || [];
  if (!perLayer.length || !hist.length) {
    return (
      <div className="mt-3 pt-3 border-t border-slate-800 text-[11px] text-slate-500">
        Gradient health appears here once training starts (take a step).
      </div>
    );
  }
  const total = hist[hist.length - 1];
  const maxLayer = Math.max(...perLayer, 1e-9);
  // Heuristic thresholds on the mean-gradient norm. These are deliberately wide;
  // the goal is to flag clearly pathological regimes, not micromanage.
  let status, statusColor;
  if (total < 1e-4) { status = 'Vanishing — gradients near zero, learning has stalled'; statusColor = 'text-amber-300'; }
  else if (total > 50) { status = 'Exploding — gradients very large, training may diverge'; statusColor = 'text-rose-300'; }
  else { status = 'Healthy range'; statusColor = 'text-emerald-300'; }
  // Sparkline geometry (log scale, since norms span orders of magnitude).
  const w = 120, h = 22;
  const logs = hist.map(v => Math.log10(Math.max(v, 1e-6)));
  const lo = Math.min(...logs), hi = Math.max(...logs);
  const span = Math.max(hi - lo, 0.5);
  const pts = logs.map((v, i) => `${(i / Math.max(1, logs.length - 1)) * w},${h - ((v - lo) / span) * h}`).join(' ');
  return (
    <div className="mt-3 pt-3 border-t border-slate-800">
      <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
        <span className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
          <Activity size={12} className="text-fuchsia-400" /> Gradient health
          <Info text="L2 norm of the mean gradient Adam receives, per layer and over time (log scale). Near-zero = vanishing (stalled learning, common with deep sigmoid/tanh stacks); very large = exploding (unstable). A healthy net sits in between and trends down as it converges." w={320} />
        </span>
        <span className={`text-[10px] ${statusColor}`}>‖∇‖ = {total < 1e-3 ? total.toExponential(1) : total.toFixed(3)} · {status}</span>
      </div>
      <div className="flex items-center gap-3">
        <svg width={w} height={h} className="shrink-0" style={{ overflow: 'visible' }}>
          <polyline points={pts} fill="none" stroke="#e879f9" strokeWidth="1.5" />
        </svg>
        <div className="flex-1 min-w-0 flex items-end gap-1" style={{ height: h }}>
          {perLayer.map((g, k) => {
            const frac = g / maxLayer;
            const isOut = k === perLayer.length - 1;
            return (
              <div key={k} className="flex-1 flex flex-col items-center justify-end" title={`Layer ${k + 1}${isOut ? ' (output)' : ''}: ‖∇‖ = ${g.toExponential(2)}`}>
                <div className="w-full rounded-t" style={{ height: `${Math.max(2, frac * h)}px`, background: isOut ? '#64748b' : '#a855f7' }} />
                <span className="text-[8px] text-slate-600 mt-0.5">{k + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function NetworkPanel({ agent, state, lastAction, syncFlash, flat }) {
  if (!agent || !state) return null;
  const net = agent.q;
  // Always recompute from the live network so depth/activation match the current
  // architecture (lastAction's cached a1/a2 may be stale across a rebuild).
  const fwd = net.forward(state);
  const layerOuts = fwd.layerOuts;          // [input(36), h0, h1, ..., q(4)]
  if (!layerOuts || layerOuts.length < 2) return null;
  const hiddenOuts = layerOuts.slice(1, layerOuts.length - 1); // hidden activations only
  const nHidden = hiddenOuts.length;
  const hiddenSpec = net.hidden;            // [{size, act}]

  const layerHeight = 240, inputBoxSize = 18, inputX = 20, inputY = 30;
  const inputGridSize = inputBoxSize * 6;
  const colGap = Math.max(64, Math.min(100, Math.floor(520 / Math.max(1, nHidden + 1))));
  const firstHX = inputX + inputGridSize + 70;
  const hiddenX = hiddenOuts.map((_, k) => firstHX + k * colGap);
  const outX = (hiddenX.length ? hiddenX[hiddenX.length - 1] : firstHX) + colGap;
  const totalW = outX + 90;

  let activeInputIdx = -1;
  for (let i = 0; i < state.length; i++) if (state[i] > 0.5) { activeInputIdx = i; break; }
  const activeY = Math.floor(activeInputIdx / W), activeX = activeInputIdx % W;

  // Per-hidden-layer geometry + top-K + max-activation for normalization.
  const layerMeta = hiddenOuts.map((act, k) => {
    const size = act.length;
    const dotR = Math.max(2, Math.min(5, Math.floor(layerHeight / size / 2)));
    const dotGap = (layerHeight - 2 * dotR) / Math.max(1, size - 1);
    let maxA = 1e-6; for (let i = 0; i < size; i++) if (Math.abs(act[i]) > maxA) maxA = Math.abs(act[i]);
    const top = topKIndices(act, Math.min(8, size));
    return { size, dotR, dotGap, maxA, top, x: hiddenX[k], act: hiddenSpec[k]?.act || 'relu' };
  });
  const yOf = (meta, i) => inputY + meta.dotR + i * meta.dotGap;

  let maxAbsQ = 1e-6, bestQ = -Infinity, bestQIdx = 0;
  for (let i = 0; i < 4; i++) { if (Math.abs(fwd.q[i]) > maxAbsQ) maxAbsQ = Math.abs(fwd.q[i]); if (fwd.q[i] > bestQ) { bestQ = fwd.q[i]; bestQIdx = i; } }
  const qBoxW = 70, qBoxH = 30, qGap = 8;
  const qBlockH = 4 * qBoxH + 3 * qGap;
  const qY0 = inputY + (layerHeight - qBlockH) / 2;
  const inSrcX = inputX + activeX * inputBoxSize + inputBoxSize / 2;
  const inSrcY = inputY + activeY * inputBoxSize + inputBoxSize / 2;

  const archStr = `36 → ${hiddenSpec.map(h => h.size).join(' → ')} → 4`;

  return (
    <div className={cardCls(flat, 'p-4', syncFlash > 0 ? 'ring-2 ring-amber-400' : '')}>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          Q-Network (live forward pass)
          <Info text="Maps the one-hot state to four Q-values through your chosen hidden layers. Adam optimizer, Huber loss against TD targets from a target network." w={300} />
        </h2>
        <div className="text-xs text-slate-400 font-mono">{archStr} · Adam · Huber</div>
      </div>
      <div className="w-full">
        <svg viewBox={`0 0 ${totalW} ${layerHeight + 80}`} width={totalW} height={layerHeight + 80}
          style={{ width: '100%', maxWidth: totalW, height: 'auto', display: 'block' }} className="mx-auto">
          <text x={inputX + inputGridSize / 2} y={20} textAnchor="middle" fontSize="11" fill="#94a3b8">Input (state)</text>
          {layerMeta.map((m, k) => (
            <text key={`hl-${k}`} x={m.x} y={20} textAnchor="middle" fontSize="10" fill="#94a3b8">
              H{k + 1} ({ACT_NAMES[m.act] || m.act})
            </text>
          ))}
          <text x={outX + 30} y={20} textAnchor="middle" fontSize="11" fill="#94a3b8">Q-values</text>

          {Array.from({ length: 36 }).map((_, i) => {
            const ix = i % W, iy = Math.floor(i / W), active = i === activeInputIdx;
            return <rect key={i} x={inputX + ix * inputBoxSize} y={inputY + iy * inputBoxSize} width={inputBoxSize - 1} height={inputBoxSize - 1} fill={active ? '#3b82f6' : '#1e293b'} stroke="#0f172a" />;
          })}
          <text x={inputX + inputGridSize / 2} y={inputY + inputGridSize + 14} textAnchor="middle" fontSize="9" fill="#64748b">one-hot, position ({activeX},{activeY})</text>

          {/* connections: input(active) -> H1 */}
          {layerMeta.length > 0 && layerMeta[0].top.map((hi, k) => {
            const op = clamp(hiddenOuts[0][hi] / layerMeta[0].maxA, 0.08, 1);
            return <line key={`ci-${k}`} x1={inSrcX} y1={inSrcY} x2={layerMeta[0].x} y2={yOf(layerMeta[0], hi)}
              stroke="#3b82f6" strokeWidth={0.6 + op * 1.4} opacity={0.15 + op * 0.55} />;
          })}

          {/* connections between consecutive hidden layers */}
          {layerMeta.slice(0, -1).map((m, k) => {
            const next = layerMeta[k + 1];
            return m.top.map((hi, ki) => next.top.map((hj, kj) => {
              const op = clamp((Math.abs(hiddenOuts[k][hi]) * Math.abs(hiddenOuts[k + 1][hj])) / (m.maxA * next.maxA), 0.05, 1);
              return <line key={`c-${k}-${ki}-${kj}`} x1={m.x} y1={yOf(m, hi)} x2={next.x} y2={yOf(next, hj)}
                stroke="#60a5fa" strokeWidth={0.5 + op * 1.2} opacity={0.1 + op * 0.45} />;
            }));
          })}

          {/* connections: last hidden -> Q outputs */}
          {layerMeta.length > 0 && (() => {
            const m = layerMeta[layerMeta.length - 1];
            return m.top.map((hj, kj) => Array.from({ length: 4 }).map((_, ai) => {
              const op = clamp(Math.abs(hiddenOuts[hiddenOuts.length - 1][hj]) / m.maxA, 0.05, 1);
              return <line key={`co-${kj}-${ai}`} x1={m.x} y1={yOf(m, hj)} x2={outX} y2={qY0 + ai * (qBoxH + qGap) + qBoxH / 2}
                stroke="#93c5fd" strokeWidth={0.4 + op * 1.0} opacity={0.08 + op * 0.4} />;
            }));
          })()}

          {/* hidden-layer neuron dots */}
          {layerMeta.map((m, k) => Array.from({ length: m.size }).map((_, i) => {
            const v = hiddenOuts[k][i];
            const t = clamp(Math.abs(v) / m.maxA, 0, 1);
            const isTop = m.top.includes(i);
            return (
              <circle key={`d-${k}-${i}`} cx={m.x} cy={yOf(m, i)} r={m.dotR + (isTop ? 1 : 0)}
                fill={v > 0 ? actColor(v) : (v < 0 ? '#7f1d1d' : '#1e293b')}
                stroke={isTop ? '#f59e0b' : '#0f172a'} strokeWidth={isTop ? 1 : 0.5}
                opacity={0.35 + t * 0.65} />
            );
          }))}

          {/* per-layer size labels */}
          {layerMeta.map((m, k) => (
            <text key={`sz-${k}`} x={m.x} y={inputY + layerHeight + 14} textAnchor="middle" fontSize="9" fill="#64748b">{m.size} units</text>
          ))}

          {/* Q-value output boxes */}
          {Array.from({ length: 4 }).map((_, ai) => {
            const q = fwd.q[ai];
            const isBest = ai === bestQIdx;
            const isPicked = lastAction && lastAction.a === ai;
            const boxY = qY0 + ai * (qBoxH + qGap);
            return (
              <g key={`q-${ai}`}>
                <rect x={outX} y={boxY} width={qBoxW} height={qBoxH}
                  fill={qColor(q, Math.max(maxAbsQ, 0.5))}
                  stroke={isPicked ? '#f59e0b' : (isBest ? '#15803d' : '#334155')}
                  strokeWidth={isPicked ? 2.5 : (isBest ? 2 : 1)} rx="3" />
                <text x={outX + 14} y={boxY + qBoxH / 2 + 5} textAnchor="middle" fontSize="16" fill="#0f172a" fontWeight="bold">{ACTION_LABELS[ai]}</text>
                <text x={outX + qBoxW - 6} y={boxY + qBoxH / 2 + 4} textAnchor="end" fontFamily="ui-monospace, Menlo, monospace" fontSize="11" fill="#0f172a">{q.toFixed(3)}</text>
              </g>
            );
          })}
          <text x={outX + qBoxW / 2} y={qY0 + qBlockH + 14} textAnchor="middle" fontSize="9" fill="#64748b">argmax = {ACTION_NAMES[bestQIdx]}</text>
        </svg>
      </div>

      <GradientHealth net={net} />

      <div className="mt-3 flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: '#3b82f6' }} />active input</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full ring-1 ring-amber-400" style={{ background: '#60a5fa' }} />top-K activations</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-2.5 rounded-sm border border-emerald-700" />argmax Q</span>
        {lastAction && <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-2.5 rounded-sm border-2 border-amber-400" />chosen action</span>}
        {syncFlash > 0 && <span className="inline-flex items-center gap-1.5 text-amber-300"><Zap size={11} /> target network just synced</span>}
      </div>
    </div>
  );
}

// ============ Slider ============
function Slider({ label, val, min, max, step, onChange, fmt, disabled }) {
  return (
    <label className={`block ${disabled ? 'opacity-40' : ''}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">{fmt ? fmt(val) : val}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={val} disabled={disabled}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full accent-emerald-500 cursor-pointer disabled:cursor-not-allowed"
      />
    </label>
  );
}

// ============ Hyperparameter panel (technical mode) ============
// One-click hyperparameter scenarios that demonstrate distinct learning regimes.
// Each preserves the network architecture and Double-DQN choice and only sets the
// learning-dynamics knobs, so users can A/B a behavior without rebuilding. The
// changes take effect live (lr / ε / γ sync to the running agent); Reset for a
// clean-slate run. Presets adapt to the algorithm (tabular tunes α, DQN tunes lr).
function ScenarioPresets({ isTabular, hp, setHp }) {
  const apply = (patch) => setHp({ ...hp, ...patch });
  // Every preset patches the SAME set of learning-dynamics keys, differing only
  // in the values. That completeness matters: when a preset patched just its own
  // headline knob (e.g. Unstable setting only lr), the untouched ε/γ knobs kept
  // whatever the previously applied scenario left behind, so an earlier preset in
  // this list still matched and won the highlight below. With every preset fully
  // specifying the same keys, at most one can match at a time.
  const base = isTabular
    ? { alpha: 0.5, gamma: 0.95, epsStart: 1.0, epsMin: 0.05, epsDecay: 0.99 }
    : { lr: 0.005, gamma: 0.95, epsStart: 1.0, epsMin: 0.05, epsDecay: 0.99 };
  const presets = [
    {
      name: 'Balanced', tip: 'Sensible defaults — steady, reliable convergence. The baseline to compare the others against.',
      patch: { ...base },
    },
    {
      name: 'Too greedy', tip: 'ε collapses fast and floors high exploitation. The agent stops exploring early and often locks into a sub-optimal route — watch policy-match plateau below 100%.',
      patch: { ...base, epsStart: 0.6, epsMin: 0.02, epsDecay: 0.9 },
    },
    {
      name: isTabular ? 'Unstable (high α)' : 'Unstable (high LR)', tip: isTabular
        ? 'α near 1 makes each update overshoot the target. The Q-values bounce instead of settling — watch the TD-error curve stay jagged.'
        : 'Learning rate 10× the default. Gradient steps overshoot; the loss curve spikes and policy-match becomes erratic — sometimes it diverges entirely.',
      patch: isTabular ? { ...base, alpha: 0.95 } : { ...base, lr: 0.05 },
    },
    {
      name: isTabular ? 'Cautious (low α)' : 'Cautious (low LR)', tip: isTabular
        ? 'Tiny α: very smooth but very slow. Needs many more episodes to reach the goal reliably.'
        : 'Learning rate 10× smaller. Smooth, stable loss but slow — it takes far longer to climb toward the optimal policy.',
      patch: isTabular ? { ...base, alpha: 0.05 } : { ...base, lr: 0.0005 },
    },
  ];
  // Mark the active preset if hp matches one. Presets are mutually exclusive by
  // construction (see above), so the first match is the only match.
  const activeName = (() => {
    for (const p of presets) {
      const keys = Object.keys(p.patch);
      if (keys.every(k => Math.abs((hp[k] ?? 0) - p.patch[k]) < 1e-9)) return p.name;
    }
    return null;
  })();
  return (
    <div className="mb-3">
      <div className="flex items-center gap-1.5 mb-1.5 text-[11px] text-slate-400">
        Scenarios
        <Info text="One-click configurations that demonstrate how hyperparameters change learning. They set the learning-rate and exploration knobs but keep your network architecture. Great paired with Compare mode: load a scenario on A, keep Balanced on B." w={320} />
      </div>
      <div className="flex flex-wrap gap-1.5">
        {presets.map(p => (
          <button key={p.name} onClick={() => apply(p.patch)} title={p.tip}
            className={`px-2 py-1 rounded-lg text-[11px] border transition ${
              activeName === p.name
                ? 'bg-emerald-600 border-emerald-500 text-white'
                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}>
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function HpPanel({ algorithm, hp, setHp, archChanged, flat }) {
  const isTabular = algorithm === 'tabular';
  const set = (k, v) => setHp({ ...hp, [k]: v });
  return (
    <div className={cardCls(flat)}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold flex items-center gap-2">
          Hyperparameters
          <Info text="Tune the learning dynamics. Changes to network architecture require a reset; the rest take effect on the next step." w={280} />
        </h2>
      </div>
      <ScenarioPresets isTabular={isTabular} hp={hp} setHp={setHp} />
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-3 text-xs">
        {isTabular ? (
          <Slider label="α (learning rate)" val={hp.alpha} min={0.01} max={1.0} step={0.01} onChange={v => set('alpha', v)} fmt={v => v.toFixed(2)} />
        ) : (
          <Slider label="Learning rate (Adam)" val={hp.lr} min={0.0005} max={0.05} step={0.0005} onChange={v => set('lr', v)} fmt={v => v.toFixed(4)} />
        )}
        <Slider label="γ (discount)" val={hp.gamma} min={0.5} max={0.99} step={0.01} onChange={v => set('gamma', v)} fmt={v => v.toFixed(2)} />
        <Slider label="ε start" val={hp.epsStart} min={0.1} max={1.0} step={0.05} onChange={v => set('epsStart', v)} fmt={v => v.toFixed(2)} />
        <Slider label="ε min" val={hp.epsMin} min={0.01} max={0.5} step={0.01} onChange={v => set('epsMin', v)} fmt={v => v.toFixed(2)} />
        <Slider label="ε decay" val={hp.epsDecay} min={0.9} max={0.999} step={0.001} onChange={v => set('epsDecay', v)} fmt={v => v.toFixed(3)} />
        {!isTabular && (
          <>
            <Slider label="Batch size" val={hp.batchSize} min={4} max={128} step={4} onChange={v => set('batchSize', Math.round(v))} fmt={v => v} />
            <Slider label="Replay capacity" val={hp.bufferSize} min={200} max={5000} step={100} onChange={v => set('bufferSize', Math.round(v))} fmt={v => v} />
            <Slider label="Target sync (steps)" val={hp.targetUpdate} min={5} max={500} step={5} onChange={v => set('targetUpdate', Math.round(v))} fmt={v => v} />
          </>
        )}
      </div>
      {!isTabular && (
        <NetworkArchitectureEditor
          layers={hp.layers || [{ size: hp.hSize, act: 'relu' }, { size: hp.hSize, act: 'relu' }]}
          setLayers={(layers) => setHp({ ...hp, layers })}
        />
      )}
      {!isTabular && (
        <label className="mt-3 flex items-center justify-between gap-2 rounded-lg bg-slate-800/50 border border-slate-700 px-3 py-2 cursor-pointer">
          <span className="text-xs text-slate-300 flex items-center gap-1.5">
            <span className="font-medium">Double DQN</span>
            <Info text="Vanilla DQN selects AND evaluates the next action with the target network — one max, which systematically overestimates Q. Double DQN selects the next action with the online network but evaluates it with the target network, cutting that maximization bias. Takes effect on the next step; no reset needed." w={320} />
          </span>
          <button onClick={() => set('doubleDQN', !hp.doubleDQN)}
            role="switch" aria-checked={!!hp.doubleDQN}
            className={`relative w-9 h-5 rounded-full transition shrink-0 ${hp.doubleDQN ? 'bg-emerald-500' : 'bg-slate-600'}`}>
            <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${hp.doubleDQN ? 'translate-x-4' : ''}`} />
          </button>
        </label>
      )}
      {archChanged && (
        <div className="mt-3 flex items-center gap-1.5 text-[11px] text-amber-300">
          <AlertTriangle size={12} /> Architecture differs from the live network — reset to rebuild it.
        </div>
      )}
    </div>
  );
}

// ============ Network architecture editor ============
// Lets the user design the hidden layers of the Q-network: add/remove layers,
// set each layer's width, and pick its activation. Input (36) and output (4) are
// fixed by the environment. Changes apply on the next Reset (the warning fires).
const MAX_LAYERS = 5;
function NetworkArchitectureEditor({ layers, setLayers }) {
  const totalParams = (() => {
    const dims = [W * Hg, ...layers.map(l => l.size), 4];
    let p = 0;
    for (let k = 0; k < dims.length - 1; k++) p += dims[k] * dims[k + 1] + dims[k + 1];
    return p;
  })();
  const updateLayer = (i, patch) => setLayers(layers.map((l, j) => j === i ? { ...l, ...patch } : l));
  const addLayer = () => { if (layers.length < MAX_LAYERS) setLayers([...layers, { size: 32, act: 'relu' }]); };
  const removeLayer = (i) => { if (layers.length > 1) setLayers(layers.filter((_, j) => j !== i)); };
  return (
    <div className="mt-3 rounded-lg bg-slate-800/40 border border-slate-700 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-200 flex items-center gap-1.5">
          <Network size={13} className="text-sky-400" /> Network architecture
          <Info text="Design the hidden layers of the Q-network. Input is the 36-cell one-hot state; output is the 4 action-values. Add up to 5 hidden layers, set each width, and choose an activation. Apply with Reset." w={320} />
        </span>
        <span className="text-[10px] font-mono text-slate-500">{totalParams.toLocaleString()} params</span>
      </div>

      {/* architecture summary chips: in → h... → out */}
      <div className="flex items-center gap-1 flex-wrap mb-3 text-[10px] font-mono">
        <span className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">36 in</span>
        {layers.map((l, i) => (
          <React.Fragment key={i}>
            <span className="text-slate-600">→</span>
            <span className="px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30">{l.size}·{ACT_NAMES[l.act] || l.act}</span>
          </React.Fragment>
        ))}
        <span className="text-slate-600">→</span>
        <span className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">4 out</span>
      </div>

      <div className="space-y-2">
        {layers.map((l, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 w-12 shrink-0">Layer {i + 1}</span>
            <input type="range" min={4} max={128} step={4} value={l.size}
              onChange={e => updateLayer(i, { size: parseInt(e.target.value, 10) })}
              className="flex-1 accent-sky-500" aria-label={`Layer ${i + 1} width`} />
            <span className="text-[11px] font-mono text-slate-300 w-8 text-right shrink-0">{l.size}</span>
            <select value={l.act} onChange={e => updateLayer(i, { act: e.target.value })}
              aria-label={`Layer ${i + 1} activation`}
              className="text-[11px] bg-slate-900 border border-slate-700 rounded px-1 py-1 text-slate-200 focus:outline-none focus:border-sky-500">
              {Object.keys(ACT_NAMES).map(k => <option key={k} value={k}>{ACT_NAMES[k]}</option>)}
            </select>
            <button onClick={() => removeLayer(i)} disabled={layers.length <= 1}
              title={layers.length <= 1 ? 'Need at least one hidden layer' : 'Remove layer'}
              className="p-1 rounded text-slate-500 hover:text-rose-300 hover:bg-rose-500/10 disabled:opacity-30 disabled:cursor-not-allowed shrink-0">
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 mt-2">
        <button onClick={addLayer} disabled={layers.length >= MAX_LAYERS}
          title={layers.length >= MAX_LAYERS ? `Max ${MAX_LAYERS} hidden layers` : 'Add a hidden layer'}
          className="px-2 py-1 text-[11px] rounded bg-slate-700 hover:bg-slate-600 text-slate-200 flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed">
          + Add layer
        </button>
        <button onClick={() => setLayers([{ size: 32, act: 'relu' }, { size: 32, act: 'relu' }])}
          className="px-2 py-1 text-[11px] rounded bg-slate-800 hover:bg-slate-700 text-slate-400">
          Reset to default
        </button>
      </div>
    </div>
  );
}

// ============ Experience-replay buffer panel (DQN only) ============
function ReplayPanel({ buffer, flat }) {
  const cap = buffer.length;
  const recent = buffer.slice(-12).reverse();
  return (
    <div className={cardCls(flat)}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold flex items-center gap-2">
          Experience replay
          <Info text="DQN stores past transitions (s, a, r, s', done) and trains on random minibatches from them. This breaks correlations between consecutive steps and reuses data." w={300} />
        </h2>
        <span className="text-xs text-slate-400 font-mono">{cap} stored</span>
      </div>
      {cap === 0 ? (
        <div className="text-xs text-slate-500 py-6 text-center">Buffer empty — take a few steps to start collecting transitions.</div>
      ) : (
        <>
          <div className="text-[10px] text-slate-500 mb-1.5">Most recent transitions (newest first)</div>
          <div className="space-y-1 font-mono text-[11px] max-h-56 overflow-y-auto pr-1">
            {recent.map((t, i) => {
              const [sx, sy] = posOfState(t.s);
              const [nx, ny] = posOfState(t.s2);
              const rPos = t.r >= 0;
              return (
                <div key={i} className="flex items-center gap-2 bg-slate-800/50 rounded px-2 py-1">
                  <span className="text-slate-300">({sx},{sy})</span>
                  <span className="text-slate-500">{ACTION_LABELS[t.a]}</span>
                  <ChevronsRight size={12} className="text-slate-600" />
                  <span className="text-slate-300">({nx},{ny})</span>
                  <span className={rPos ? 'text-emerald-400' : 'text-rose-400'}>
                    {rPos ? '+' : ''}{t.r.toFixed(2)}
                  </span>
                  {t.terminal && <span className="text-amber-400">done</span>}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
const posOfState = (s) => {
  for (let i = 0; i < s.length; i++) if (s[i] > 0.5) return [i % W, Math.floor(i / W)];
  return [-1, -1];
};

// ============ Q-table panel (tabular only) ============
function QTablePanel({ grid, qMap, optimalV, optimalPolicy, flat: flatChrome }) {
  const rows = [];
  if (qMap?.cells) {
    for (let y = 0; y < Hg; y++) {
      for (let x = 0; x < W; x++) {
        const cell = qMap.cells[y]?.[x];
        if (!cell || grid[y][x] === 1) continue;
        const q = cell.q;
        let best = 0;
        for (let i = 1; i < 4; i++) if (q[i] > q[best]) best = i;
        const idx = y * W + x;
        const opt = optimalPolicy ? optimalPolicy[idx] : null;
        const agrees = opt == null || opt < 0 || opt === best;
        const vstar = optimalV ? optimalV[idx] : null;
        rows.push({ x, y, q, best, opt, agrees, vstar });
      }
    }
  }
  return (
    <div className={cardCls(flatChrome)}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold flex items-center gap-2">
          Q-table
          <Info text="Tabular Q-learning stores one row of four Q-values per state — a literal lookup table, no generalization. The arrow marks the greedy action; red means it disagrees with the optimal policy." w={300} />
        </h2>
        <span className="text-xs text-slate-400 font-mono">{rows.length} states</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-slate-400 text-left">
              <th className="py-1 pr-3 font-medium">State</th>
              <th className="py-1 px-2 font-medium text-center">↑</th>
              <th className="py-1 px-2 font-medium text-center">→</th>
              <th className="py-1 px-2 font-medium text-center">↓</th>
              <th className="py-1 px-2 font-medium text-center">←</th>
              <th className="py-1 px-2 font-medium text-center">greedy</th>
              <th className="py-1 pl-2 font-medium text-right">V*</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-slate-800/60">
                <td className="py-1 pr-3 text-slate-300">({r.x},{r.y})</td>
                {r.q.map((v, ai) => (
                  <td key={ai} className={`py-1 px-2 text-center ${ai === r.best ? 'text-slate-100 font-semibold' : 'text-slate-500'}`}>
                    {v.toFixed(2)}
                  </td>
                ))}
                <td className="py-1 px-2 text-center">
                  <span className={r.agrees ? 'text-emerald-400' : 'text-rose-400'}>{ACTION_LABELS[r.best]}</span>
                </td>
                <td className="py-1 pl-2 text-right text-slate-400">{r.vstar != null ? r.vstar.toFixed(2) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="text-xs text-slate-500 py-6 text-center">No visited states yet — take a few steps.</div>
        )}
      </div>
    </div>
  );
}

// ============ Beginner explainer (simple mode) ============
function BeginnerExplain({ algorithm, onStartTour, onStartMath }) {
  const isTabular = algorithm === 'tabular';
  const cards = [
    {
      icon: <Target size={18} className="text-emerald-400" />,
      title: 'The goal',
      body: 'The blue dot wants to reach the 🚩 flag while avoiding 💀 hazards. Every step costs a little, so it learns to find short, safe paths.',
    },
    {
      icon: <Sparkles size={18} className="text-amber-400" />,
      title: 'Trial and error',
      body: 'At first it moves almost randomly (exploring). Good moves earn reward, bad moves lose it. Over many tries it remembers which directions pay off from each square.',
    },
    {
      icon: isTabular
        ? <Table2 size={18} className="text-sky-400" />
        : <Brain size={18} className="text-sky-400" />,
      title: isTabular ? 'A memory table' : 'A tiny brain',
      body: isTabular
        ? 'Tabular Q-learning keeps a notebook with a score for each direction in each square, and slowly updates those scores as it learns.'
        : 'Deep Q-learning uses a small neural network to guess the score for each direction — so it can generalize across squares instead of memorizing each one.',
    },
    {
      icon: <Route size={18} className="text-fuchsia-400" />,
      title: 'Watch it converge',
      body: 'As the colored arrows stop changing, the agent has settled on a policy. Hit Rollout to watch it run the learned path with no more guessing.',
    },
  ];
  return (
    <div className="mt-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          What&apos;s happening here?
        </h2>
        <button
          onClick={onStartTour}
          className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 text-sm">
          Take the tour
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-3 mb-3 pb-3 border-b border-slate-800/60">
        <div className="text-xs text-slate-400 flex-1 min-w-[160px]">
          Prefer the theory first? Walk through the equations behind it all.
        </div>
        <button
          onClick={onStartMath}
          className="px-3 py-1.5 rounded-lg bg-violet-600/20 border border-violet-500/50 text-violet-200 hover:bg-violet-600/30 flex items-center gap-1.5 text-sm whitespace-nowrap">
          The math behind it
        </button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {cards.map((c, i) => (
          <div key={i} className="bg-slate-800/50 rounded-lg p-3 border border-slate-800">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="font-medium text-slate-200 text-sm">{c.title}</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{c.body}</p>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] text-slate-500">
        Switch to <span className="text-slate-300">Technical</span> mode (top right) to see the Bellman update, replay buffer, and live network.
      </p>
    </div>
  );
}

// ============ Footer ============
function Footer({ algorithm, doubleDQN }) {
  return (
    <footer className="mt-6 pt-4 border-t border-slate-800 text-[11px] text-slate-500 flex flex-wrap items-center justify-between gap-2">
      <span>
        {algorithm === 'tabular' ? 'Tabular Q-learning' : (doubleDQN ? 'Double DQN' : 'Deep Q-Network (DQN)')} · 6×6 grid world · γ-discounted, ε-greedy
      </span>
      <span className="flex items-center gap-1.5">
        <Network size={12} /> An interactive reinforcement-learning sandbox
      </span>
    </footer>
  );
}

// ============ Math lesson ============
// Lightweight, self-contained math typesetting (no external KaTeX dependency).
// Variables render italic-serif; operators stay upright; accent colors mark the
// quantity in focus for each equation.
function V({ children }) { return <i className="ml-var">{children}</i>; }            // variable
function Op({ children }) { return <span className="ml-op">{children}</span>; }       // operator / function name
const C = {                                                                          // accent colors
  r: '#34d399', g: '#fbbf24', q: '#60a5fa', td: '#f472b6', th: '#a78bfa', s: '#94a3b8',
};
function Tok({ c, children }) { return <span style={{ color: c }}>{children}</span>; }

const MATH_LESSON = [
  {
    title: 'The reinforcement learning loop',
    eq: (
      <span>
        <Op>agent</Op> &nbsp;—<Tok c={C.q}>action <V>a</V><sub>t</sub></Tok>→&nbsp; <Op>environment</Op>
        &nbsp;—<Tok c={C.r}>reward <V>r</V><sub>t+1</sub></Tok>, <Tok c={C.s}>state <V>s</V><sub>t+1</sub></Tok>→&nbsp; <Op>agent</Op>
      </span>
    ),
    body: 'At each tick the agent observes a state, picks an action, and the environment returns a reward and the next state. That single transition — (s, a, r, s′) — is the only thing the agent ever learns from.',
    tie: 'Each Step button click in the grid is exactly one turn of this loop.',
  },
  {
    title: 'Formalizing it: a Markov Decision Process',
    eq: (
      <span>
        ⟨ <V>S</V>, <V>A</V>, <V>P</V>(<V>s</V>′ | <V>s</V>, <V>a</V>), <Tok c={C.r}><V>R</V>(<V>s</V>, <V>a</V>)</Tok>, <Tok c={C.g}><V>γ</V></Tok> ⟩
      </span>
    ),
    body: 'States S, actions A, transition probabilities P, a reward function R, and a discount γ ∈ [0,1). "Markov" means the next state depends only on the current state and action — not the full history. The grid world here is a deterministic MDP: moving is certain, walls block, the flag and hazards end the episode.',
    tie: 'S = the 36 cells, A = {↑ → ↓ ←}, R = −0.04 per step, +1 at the flag, −1 at a hazard.',
  },
  {
    title: 'The goal: maximize discounted return',
    eq: (
      <div className="space-y-1">
        <div><V>G</V><sub>t</sub> = <Tok c={C.r}><V>r</V><sub>t+1</sub></Tok> + <Tok c={C.g}><V>γ</V></Tok> <Tok c={C.r}><V>r</V><sub>t+2</sub></Tok> + <Tok c={C.g}><V>γ</V></Tok><sup>2</sup> <Tok c={C.r}><V>r</V><sub>t+3</sub></Tok> + ⋯</div>
        <div className="text-base opacity-80">= <span className="ml-bigop">∑</span><sub className="align-baseline">k=0</sub><sup>∞</sup> <Tok c={C.g}><V>γ</V></Tok><sup>k</sup> <Tok c={C.r}><V>r</V><sub>t+k+1</sub></Tok></div>
      </div>
    ),
    body: 'The agent does not maximize the next reward — it maximizes the whole future, discounted. γ < 1 makes near-term reward worth more than distant reward and keeps the sum finite. A small γ is short-sighted; a γ near 1 plans far ahead.',
    tie: 'Lower γ in the hyperparameters and watch value stop propagating back from the flag along long corridors.',
  },
  {
    title: 'Policies and value functions',
    eq: (
      <div className="space-y-2">
        <div><Tok c={C.q}><V>V</V><sup>π</sup>(<V>s</V>)</Tok> = 𝔼<sub>π</sub>[ <V>G</V><sub>t</sub> | <V>s</V><sub>t</sub> = <V>s</V> ]</div>
        <div><Tok c={C.q}><V>Q</V><sup>π</sup>(<V>s</V>, <V>a</V>)</Tok> = 𝔼<sub>π</sub>[ <V>G</V><sub>t</sub> | <V>s</V><sub>t</sub> = <V>s</V>, <V>a</V><sub>t</sub> = <V>a</V> ]</div>
      </div>
    ),
    body: 'A policy π says which action to take in each state. Its state-value V^π is the expected return from a state; its action-value Q^π is the expected return from taking a specific action first, then following π. Q is what we actually learn here — one number per (state, action).',
    tie: 'The four wedges in every grid cell are the four Q(s, a) values; the center arrow is argmax over them.',
  },
  {
    title: 'The Bellman equation: value is recursive',
    eq: (
      <span>
        <Tok c={C.q}><V>Q</V><sup>π</sup>(<V>s</V>, <V>a</V>)</Tok> = 𝔼[ <Tok c={C.r}><V>r</V></Tok> + <Tok c={C.g}><V>γ</V></Tok> <Tok c={C.q}><V>Q</V><sup>π</sup>(<V>s</V>′, <V>a</V>′)</Tok> ]
      </span>
    ),
    body: 'Here is the key insight that makes learning possible: the value of acting now equals the immediate reward plus the discounted value of where you land. Long-horizon value folds into a one-step relationship — you never need to simulate the entire future, only look one step ahead and trust your own estimate of the rest.',
    tie: 'This "trust your next estimate" is bootstrapping — the engine behind every update in the app.',
  },
  {
    title: 'Optimality: the best possible Q',
    eq: (
      <div className="space-y-2">
        <div><Tok c={C.q}><V>Q</V>*(<V>s</V>, <V>a</V>)</Tok> = 𝔼[ <Tok c={C.r}><V>r</V></Tok> + <Tok c={C.g}><V>γ</V></Tok> <Op>max</Op><sub><V>a</V>′</sub> <Tok c={C.q}><V>Q</V>*(<V>s</V>′, <V>a</V>′)</Tok> ]</div>
        <div className="text-base opacity-80">π*(<V>s</V>) = <Op>argmax</Op><sub><V>a</V></sub> <Tok c={C.q}><V>Q</V>*(<V>s</V>, <V>a</V>)</Tok></div>
      </div>
    ),
    body: 'Swap the policy\u2019s expectation for a max over next actions and you get the Bellman optimality equation: the value of always acting optimally. Once you know Q*, the optimal policy is trivial — in each state, pick the action with the highest Q*. Solving this exactly (when P and R are known) is value iteration.',
    tie: 'The V* / optimal-arrow overlay is precisely this Q*, computed by value iteration as ground truth.',
  },
  {
    title: 'Q-learning: estimating Q* from samples',
    eq: (
      <div className="space-y-2">
        <div>
          <Tok c={C.q}><V>Q</V>(<V>s</V>, <V>a</V>)</Tok> ← <Tok c={C.q}><V>Q</V>(<V>s</V>, <V>a</V>)</Tok> + <Tok c={C.g}><V>α</V></Tok>
          &nbsp;[ <Tok c={C.td}><V>r</V> + <V>γ</V> <Op>max</Op><sub><V>a</V>′</sub> <V>Q</V>(<V>s</V>′, <V>a</V>′) − <V>Q</V>(<V>s</V>, <V>a</V>)</Tok> ]
        </div>
        <div className="text-base opacity-80">
          <Tok c={C.td}>δ</Tok> = <span className="ml-op">TD error</span> = target − current estimate
        </div>
      </div>
    ),
    body: 'We do not know P or R, so we cannot solve Bellman directly — instead we sample transitions and nudge. The bracket is the temporal-difference error δ: how wrong the current estimate was versus the one-step target r + γ·max Q(s′,·). Move Q a fraction α toward the target. Repeat over many transitions and Q drifts toward Q*.',
    tie: 'The Bellman panel shows exactly this: target, current Q, δ, and the updated value after each step.',
  },
  {
    title: 'Exploration vs exploitation: ε-greedy',
    eq: (
      <span>
        <V>a</V><sub>t</sub> = <span className="inline-flex flex-col items-start align-middle text-base ml-cases">
          <span><Op>argmax</Op><sub><V>a</V></sub> <Tok c={C.q}><V>Q</V>(<V>s</V>, <V>a</V>)</Tok> &nbsp;<span className="opacity-70">with prob. 1 − <Tok c={C.g}><V>ε</V></Tok></span></span>
          <span>random action &nbsp;<span className="opacity-70">with prob. <Tok c={C.g}><V>ε</V></Tok></span></span>
        </span>
      </span>
    ),
    body: 'A purely greedy agent gets stuck exploiting whatever it found first and never discovers better routes. ε-greedy forces occasional random actions so every state-action pair keeps getting sampled. ε starts high (explore) and decays toward a small floor (exploit) as the estimates mature.',
    tie: 'Orange agent pulses = an exploratory move; the ε curve in technical mode shows the decay.',
  },
  {
    title: 'Scaling up: Deep Q-Networks',
    eq: (
      <div className="space-y-2">
        <div><span className="ml-op">L</span>(<Tok c={C.th}>θ</Tok>) = 𝔼[ ( <V>y</V> − <Tok c={C.q}><V>Q</V>(<V>s</V>, <V>a</V>; <Tok c={C.th}>θ</Tok>)</Tok> )<sup>2</sup> ]</div>
        <div className="text-base opacity-80"><V>y</V> = <Tok c={C.r}><V>r</V></Tok> + <Tok c={C.g}><V>γ</V></Tok> <Op>max</Op><sub><V>a</V>′</sub> <V>Q</V>(<V>s</V>′, <V>a</V>′; <Tok c={C.th}>θ</Tok><sup>−</sup>)</div>
      </div>
    ),
    body: 'A lookup table cannot scale to large or continuous state spaces. DQN replaces it with a neural network Q(s,a;θ) and turns the TD update into gradient descent on a squared-error loss against the target y. Two stabilizers matter: a target network θ⁻ (a slowly-synced copy that keeps y from chasing itself) and experience replay (training on random past transitions to break correlation).',
    tie: 'The live network panel is Q(·;θ); the buffer panel is replay; the amber flash is a θ⁻ sync.',
  },
  {
    title: 'Overestimation, and Double DQN',
    eq: (
      <div className="space-y-2">
        <div className="text-base opacity-80"><Op>vanilla</Op>: <V>y</V> = <Tok c={C.r}><V>r</V></Tok> + <Tok c={C.g}><V>γ</V></Tok> <Op>max</Op><sub><V>a</V>′</sub> <V>Q</V>(<V>s</V>′, <V>a</V>′; <Tok c={C.th}>θ</Tok><sup>−</sup>)</div>
        <div><Op>double</Op>: <V>y</V> = <Tok c={C.r}><V>r</V></Tok> + <Tok c={C.g}><V>γ</V></Tok> <V>Q</V>(<V>s</V>′, <Op>argmax</Op><sub><V>a</V>′</sub> <V>Q</V>(<V>s</V>′,<V>a</V>′;<Tok c={C.th}>θ</Tok>); <Tok c={C.th}>θ</Tok><sup>−</sup>)</div>
      </div>
    ),
    body: 'The max in the vanilla target both PICKS and SCORES the next action with the same network, so any noise that happens to inflate one action gets selected and propagated — a systematic overestimation of Q. Double DQN splits the two: the online network θ picks the next action, the target network θ⁻ scores it. The picker and the scorer disagree on noise, so the bias largely cancels. It is a one-line change with no extra network.',
    tie: 'Toggle "Double DQN" in Hyperparameters; the Bellman panel rewrites the target to show the online-pick / target-score split. Compare mode can run vanilla vs Double side by side.',
  },
  {
    title: 'Why it converges',
    eq: (
      <span>
        <Tok c={C.q}><V>Q</V></Tok> &nbsp;→&nbsp; <Tok c={C.q}><V>Q</V>*</Tok> &nbsp;⟹&nbsp; greedy policy &nbsp;→&nbsp; π*
      </span>
    ),
    body: 'With enough exploration and a decaying learning rate, tabular Q-learning is proven to converge to Q*; DQN approximates the same fixed point. As Q approaches Q*, the greedy arrows stop changing and start matching the optimal policy. That is convergence — and it is measurable.',
    tie: 'Policy match → 100% is the agent reaching π*. Hit Rollout to watch it run the solved path.',
  },
];

function MathLesson({ step, setStep, onClose }) {
  const total = MATH_LESSON.length;
  const isLast = step === total - 1;
  const isFirst = step === 0;
  const s = MATH_LESSON[step];

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); isLast ? onClose() : setStep(step + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); if (!isFirst) setStep(step - 1); }
      else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, isLast, isFirst, onClose, setStep]);

  return (
    <>
      <style>{`
        @keyframes ml-overlay-in { from { opacity: 0; } to { opacity: 1; } }
        @keyframes ml-card-in { from { opacity: 0; transform: translateY(10px) scale(0.985); } to { opacity: 1; transform: none; } }
        @keyframes ml-step-in { from { opacity: 0; transform: translateX(10px); } to { opacity: 1; transform: none; } }
        .ml-overlay { animation: ml-overlay-in 0.2s ease-out both; }
        .ml-card { animation: ml-card-in 0.28s cubic-bezier(0.2,0.8,0.2,1) both; }
        .ml-step { animation: ml-step-in 0.28s cubic-bezier(0.3,0,0.2,1) both; }
        .ml-var { font-style: italic; }
        .ml-op { font-style: normal; }
        .ml-eq { font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, 'Times New Roman', serif; }
        .ml-eq sub, .ml-eq sup { font-style: normal; }
        .ml-bigop { font-size: 1.5em; line-height: 0; vertical-align: -0.15em; }
        .ml-cases { border-left: 2px solid #475569; padding-left: 0.6rem; gap: 0.15rem; }
        .ml-progress::-webkit-scrollbar { display: none; }
      `}</style>

      <div className="ml-overlay fixed inset-0 z-[150] bg-slate-950/80 backdrop-blur-sm" onClick={onClose} />

      <div className="fixed inset-0 z-[160] flex items-center justify-center p-4 pointer-events-none">
        <div className="ml-card pointer-events-auto w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl flex flex-col max-h-[90vh]"
          onClick={(e) => e.stopPropagation()}>

          {/* header */}
          <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-violet-500/15 border border-violet-500/40 flex items-center justify-center">
                <Sparkles size={16} className="text-violet-300" />
              </div>
              <div>
                <div className="text-sm font-semibold text-violet-200 leading-tight">The math of Q-learning</div>
                <div className="text-[10px] font-mono text-slate-500 tracking-wide">STEP {step + 1} OF {total}</div>
              </div>
            </div>
            <button onClick={onClose} className="text-slate-500 hover:text-slate-200 transition" aria-label="Close lesson">
              <X size={16} />
            </button>
          </div>

          {/* body (keyed → re-animates each step) */}
          <div key={step} className="ml-step px-6 py-5 overflow-y-auto">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">{s.title}</h3>
            <div className="ml-eq text-center text-xl sm:text-2xl text-slate-100 bg-slate-950/60 border border-slate-800 rounded-xl px-4 sm:px-6 py-5 mb-4 overflow-x-auto leading-relaxed">
              {s.eq}
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">{s.body}</p>
            {s.tie && (
              <div className="mt-3 flex items-start gap-2 text-xs text-emerald-300/90 bg-emerald-500/5 border border-emerald-500/20 rounded-lg px-3 py-2">
                <Eye size={14} className="mt-0.5 shrink-0 text-emerald-400" />
                <span>{s.tie}</span>
              </div>
            )}
          </div>

          {/* footer / nav */}
          <div className="px-6 py-4 border-t border-slate-800">
            <div className="flex items-center gap-2">
              <button onClick={onClose} className="text-xs text-slate-500 hover:text-slate-300 mr-auto">
                Close lesson
              </button>
              <button onClick={() => setStep(step - 1)} disabled={isFirst}
                className="px-3 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1">
                <ChevronLeft size={12} /> Back
              </button>
              <button onClick={() => isLast ? onClose() : setStep(step + 1)}
                className="px-3 py-1.5 text-xs rounded-lg bg-violet-600 hover:bg-violet-500 text-white flex items-center gap-1 font-medium">
                {isLast ? 'Finish' : <>Next <ChevronRight size={12} /></>}
              </button>
            </div>
            <div className="ml-progress flex gap-0.5 mt-3 overflow-x-auto">
              {MATH_LESSON.map((_, i) => (
                <button key={i} onClick={() => setStep(i)}
                  className={`h-1 flex-1 min-w-[10px] rounded transition ${i === step ? 'bg-violet-500' : i < step ? 'bg-violet-700' : 'bg-slate-700 hover:bg-slate-600'}`}
                  aria-label={`Go to step ${i + 1}`} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ============ Mount ============
// index.html loads this file through Babel Standalone as an ES module; the bare
// specifiers above ('react', 'recharts', 'lucide-react') resolve via the import
// map declared there. Everything below is the only addition to the component as
// it was authored — it just attaches <App /> to the page.
import { createRoot } from 'react-dom/client';

const rootEl = document.getElementById('root');
document.getElementById('boot')?.remove();
createRoot(rootEl).render(<App />);
