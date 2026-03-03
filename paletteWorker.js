const MODEL_CONFIG = {
  NUM_BINS_L: 42,
  NUM_BINS_C: 42,
  NUM_BINS_H: 42,
  L_BASE: 0,
  C_BASE: 42,
  H_BASE: 84,
  BOS_TOKEN: 126,
  SEP_TOKEN: 127,
  VOCAB_SIZE: 128,
  BLOCK_SIZE: 16,
  N_LAYER: 1,
  N_EMBD: 16,
  N_HEAD: 4,
  HEAD_DIM: 4,
  C_MAX: 0.21791855751774927,
};

const MLP_HIDDEN = MODEL_CONFIG.N_EMBD * 4;
const MIN_MUTED_CONTRAST = 3.0;

const TEMPERATURE_PRESETS = {
  low: { first: 0.3, middle: 0.2, last: 0.1 },
  medium: { first: 0.85, middle: 0.77, last: 0.766 },
  high: { first: 1.0, middle: 0.92, last: 0.9 },
};

const RETRO_HUES = [20, 38, 52, 92, 180, 215, 330].map((d) => d / 360);
const WEBSAFE_STEPS = [0, 51, 102, 153, 204, 255];

const MODE_RULES = {
  normal: { maxAttempts: 3000, minVibrant: 0 },
  vibrant: { maxAttempts: 10000, minVibrant: 2 },
  veryVibrant: { maxAttempts: 10000, minVibrant: 3 },
  pastel: { maxAttempts: 20000, minVibrant: 0, pastel: true },
  retro: { maxAttempts: 10000, minVibrant: 1, retro: true },
  computer90s: { maxAttempts: 3000, minVibrant: 0, computer90s: true },
  extremeComputer90s: { maxAttempts: 3000, minVibrant: 0, extremeComputer90s: true },
  grey: { maxAttempts: 3000, minVibrant: 0, grey: true },
};

let runtime = null;

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function quantizeToBins(x, lo, hi, bins) {
  const xc = Math.max(lo, Math.min(hi, x));
  const scaled = (xc - lo) / Math.max(hi - lo, 1e-12);
  return Math.max(0, Math.min(bins - 1, Math.round(scaled * (bins - 1))));
}

function dequantizeFromBins(i, lo, hi, bins) {
  return lo + (i / (bins - 1)) * (hi - lo);
}

function softmax(logits, temperature) {
  const t = Math.max(temperature, 1e-6);
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i] / t;
    if (v > maxVal) maxVal = v;
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const e = Math.exp(logits[i] / t - maxVal);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i += 1) exps[i] /= sum;
  return exps;
}

function sampleCategorical(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i += 1) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function expectedTokenTypeAtPosition(cfg, position) {
  // Sequence format learned during training:
  // [BOS, L, C, H, SEP, L, C, H, SEP, L, C, H, SEP, L, C, H, BOS]
  if (position <= 0) return "BOS";
  if (position === cfg.BLOCK_SIZE) return "BOS";

  const inPalette = position - 1; // shift so first palette token starts at 0
  const slot = inPalette % 4;
  if (slot === 0) return "L";
  if (slot === 1) return "C";
  if (slot === 2) return "H";
  return "SEP";
}

function isTokenAllowedAtPosition(cfg, tokenId, position) {
  const tokenType = expectedTokenTypeAtPosition(cfg, position);
  if (tokenType === "L") return tokenId >= cfg.L_BASE && tokenId < cfg.C_BASE;
  if (tokenType === "C") return tokenId >= cfg.C_BASE && tokenId < cfg.H_BASE;
  if (tokenType === "H") return tokenId >= cfg.H_BASE && tokenId < cfg.BOS_TOKEN;
  if (tokenType === "SEP") return tokenId === cfg.SEP_TOKEN;
  return tokenId === cfg.BOS_TOKEN;
}

function sampleCategoricalConstrained(rt, logits, temperature, nextPosition) {
  const cfg = rt.cfg;
  const t = Math.max(temperature, 1e-6);
  const allowed = [];

  for (let tokenId = 0; tokenId < cfg.VOCAB_SIZE; tokenId += 1) {
    if (isTokenAllowedAtPosition(cfg, tokenId, nextPosition)) allowed.push(tokenId);
  }

  if (allowed.length === 0) throw new Error(`No allowed tokens at position ${nextPosition}.`);
  if (allowed.length === 1) return allowed[0];

  let maxVal = -Infinity;
  for (let i = 0; i < allowed.length; i += 1) {
    const v = logits[allowed[i]] / t;
    if (v > maxVal) maxVal = v;
  }

  const probs = new Float32Array(allowed.length);
  let sum = 0;
  for (let i = 0; i < allowed.length; i += 1) {
    const e = Math.exp(logits[allowed[i]] / t - maxVal);
    probs[i] = e;
    sum += e;
  }

  let r = Math.random() * sum;
  for (let i = 0; i < allowed.length; i += 1) {
    r -= probs[i];
    if (r <= 0) return allowed[i];
  }

  return allowed[allowed.length - 1];
}

function isPrefixTokenizerValid(cfg, prefix) {
  if (!prefix || prefix.length === 0) return true;
  for (let pos = 0; pos < prefix.length; pos += 1) {
    if (!isTokenAllowedAtPosition(cfg, prefix[pos], pos)) return false;
  }
  return true;
}

function rmsnorm(x) {
  let ms = 0;
  for (let i = 0; i < x.length; i += 1) ms += x[i] * x[i];
  ms /= x.length;
  const scale = 1 / Math.sqrt(ms + 1e-5);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) out[i] = x[i] * scale;
  return out;
}

function matVec(matRows, x) {
  const out = new Float32Array(matRows.length);
  for (let r = 0; r < matRows.length; r += 1) {
    const row = matRows[r];
    let s = 0;
    for (let c = 0; c < x.length; c += 1) s += row[c] * x[c];
    out[r] = s;
  }
  return out;
}

function addVec(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] + b[i];
  return out;
}

function relu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) out[i] = Math.max(0, x[i]);
  return out;
}

function takeMatrix(flat, state, rows, cols) {
  const out = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const start = state.offset;
    const end = start + cols;
    if (end > flat.length) throw new Error("Binary weights ended early while decoding matrix.");
    out[r] = flat.subarray(start, end);
    state.offset = end;
  }
  return out;
}

function buildRuntimeFromFlat(flat) {
  const cfg = MODEL_CONFIG;
  const state = { offset: 0 };

  const wte = takeMatrix(flat, state, cfg.VOCAB_SIZE, cfg.N_EMBD);
  const wpe = takeMatrix(flat, state, cfg.BLOCK_SIZE, cfg.N_EMBD);
  const lmHead = takeMatrix(flat, state, cfg.VOCAB_SIZE, cfg.N_EMBD);

  const layers = [];
  for (let i = 0; i < cfg.N_LAYER; i += 1) {
    layers.push({
      attnWq: takeMatrix(flat, state, cfg.N_EMBD, cfg.N_EMBD),
      attnWk: takeMatrix(flat, state, cfg.N_EMBD, cfg.N_EMBD),
      attnWv: takeMatrix(flat, state, cfg.N_EMBD, cfg.N_EMBD),
      attnWo: takeMatrix(flat, state, cfg.N_EMBD, cfg.N_EMBD),
      mlpFc1: takeMatrix(flat, state, MLP_HIDDEN, cfg.N_EMBD),
      mlpFc2: takeMatrix(flat, state, cfg.N_EMBD, MLP_HIDDEN),
    });
  }

  if (state.offset !== flat.length) {
    throw new Error(`Binary weights length mismatch: consumed ${state.offset} floats, got ${flat.length}.`);
  }

  return { cfg, wte, wpe, lmHead, layers };
}

function float16ToFloat32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;

  if (e === 0) {
    if (f === 0) return s ? -0 : 0;
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }

  if (e === 31) {
    if (f === 0) return s ? -Infinity : Infinity;
    return NaN;
  }

  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function loadBinaryWeightsFromBase64(base64) {
  const buffer = base64ToArrayBuffer(base64);
  if (buffer.byteLength % 2 !== 0) {
    throw new Error(`Binary model byte length (${buffer.byteLength}) is not aligned to Float16.`);
  }

  if (typeof Float16Array !== "undefined") return new Float32Array(new Float16Array(buffer));

  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 1) out[i] = float16ToFloat32(src[i]);
  return out;
}

async function loadBinaryWeightsFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch binary model: ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength % 2 !== 0) {
    throw new Error(`Binary model byte length (${buffer.byteLength}) is not aligned to Float16.`);
  }

  if (typeof Float16Array !== "undefined") return new Float32Array(new Float16Array(buffer));

  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 1) out[i] = float16ToFloat32(src[i]);
  return out;
}

function initCache(rt) {
  const { N_LAYER, BLOCK_SIZE, N_EMBD } = rt.cfg;
  const cache = [];
  for (let i = 0; i < N_LAYER; i += 1) {
    cache.push({
      k: new Float32Array(BLOCK_SIZE * N_EMBD),
      v: new Float32Array(BLOCK_SIZE * N_EMBD),
      t: 0,
    });
  }
  return cache;
}

function gptStepFast(rt, tokenId, posId, cache) {
  const cfg = rt.cfg;
  let x = addVec(rt.wte[tokenId], rt.wpe[posId]);
  x = rmsnorm(x);

  for (let li = 0; li < cfg.N_LAYER; li += 1) {
    const layer = rt.layers[li];
    let xRes = x;
    x = rmsnorm(x);

    const q = matVec(layer.attnWq, x);
    const k = matVec(layer.attnWk, x);
    const v = matVec(layer.attnWv, x);

    const lc = cache[li];
    const t = lc.t;
    for (let i = 0; i < cfg.N_EMBD; i += 1) {
      lc.k[t * cfg.N_EMBD + i] = k[i];
      lc.v[t * cfg.N_EMBD + i] = v[i];
    }
    lc.t += 1;

    const xAttn = new Float32Array(cfg.N_EMBD);
    for (let h = 0; h < cfg.N_HEAD; h += 1) {
      const hs = h * cfg.HEAD_DIM;
      const logits = new Float32Array(t + 1);
      for (let tt = 0; tt <= t; tt += 1) {
        let dot = 0;
        const base = tt * cfg.N_EMBD + hs;
        for (let j = 0; j < cfg.HEAD_DIM; j += 1) dot += q[hs + j] * lc.k[base + j];
        logits[tt] = dot / Math.sqrt(cfg.HEAD_DIM);
      }

      const weights = softmax(logits, 1.0);
      const outH = new Float32Array(cfg.HEAD_DIM);
      for (let tt = 0; tt <= t; tt += 1) {
        const w = weights[tt];
        const base = tt * cfg.N_EMBD + hs;
        for (let j = 0; j < cfg.HEAD_DIM; j += 1) outH[j] += w * lc.v[base + j];
      }

      for (let j = 0; j < cfg.HEAD_DIM; j += 1) xAttn[hs + j] = outH[j];
    }

    x = matVec(layer.attnWo, xAttn);
    x = addVec(x, xRes);

    xRes = x;
    x = rmsnorm(x);
    x = matVec(layer.mlpFc1, x);
    x = relu(x);
    x = matVec(layer.mlpFc2, x);
    x = addVec(x, xRes);
  }

  return matVec(rt.lmHead, x);
}

function oklchColorToTokens(rt, color) {
  const cfg = rt.cfg;
  const lBin = quantizeToBins(color[0], 0, 1, cfg.NUM_BINS_L);
  const cBin = quantizeToBins(color[1], 0, cfg.C_MAX, cfg.NUM_BINS_C);
  const hBin = quantizeToBins(color[2], 0, 1, cfg.NUM_BINS_H);
  return [cfg.L_BASE + lBin, cfg.C_BASE + cBin, cfg.H_BASE + hBin];
}

function temperatureFromProgress(rt, seq, tFirst, tMiddle, tLast) {
  const sep = rt.cfg.SEP_TOKEN;
  let completed = 0;
  for (let i = 0; i < seq.length; i += 1) if (seq[i] === sep) completed += 1;
  if (completed === 0) return tFirst;
  if (completed >= 3) return tLast;
  return tMiddle;
}

function sampleFromPrefixFastDynamic(rt, prefix, tFirst, tMiddle, tLast) {
  const cfg = rt.cfg;
  const cache = initCache(rt);
  const seq = prefix.slice();

  if (seq.length === 0) return [cfg.BOS_TOKEN];
  if (!isPrefixTokenizerValid(cfg, seq)) {
    throw new Error("Invalid prefix: token order does not match tokenizer grammar.");
  }

  let logits = null;
  for (let pos = 0; pos < seq.length; pos += 1) {
    if (pos >= cfg.BLOCK_SIZE) return seq;
    logits = gptStepFast(rt, seq[pos], pos, cache);
  }

  while (seq.length <= cfg.BLOCK_SIZE) {
    const temp = temperatureFromProgress(rt, seq, tFirst, tMiddle, tLast);
    const nextPosition = seq.length;
    const tokenId = sampleCategoricalConstrained(rt, logits, temp, nextPosition);
    seq.push(tokenId);

    if (tokenId === cfg.BOS_TOKEN) break;

    const pos = seq.length - 1;
    if (pos >= cfg.BLOCK_SIZE) break;
    logits = gptStepFast(rt, tokenId, pos, cache);
  }

  return seq;
}

function stripSpecial(rt, seq) {
  return seq.filter((t) => t !== rt.cfg.BOS_TOKEN);
}

function decodePaletteTokens(rt, tokens) {
  const cfg = rt.cfg;
  const lVals = [];
  const cVals = [];
  const hVals = [];

  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (t >= cfg.L_BASE && t < cfg.C_BASE) lVals.push(t - cfg.L_BASE);
    else if (t >= cfg.C_BASE && t < cfg.H_BASE) cVals.push(t - cfg.C_BASE);
    else if (t >= cfg.H_BASE && t < cfg.BOS_TOKEN) hVals.push(t - cfg.H_BASE);
  }

  const n = Math.min(lVals.length, cVals.length, hVals.length, 4);
  if (n === 0) return null;

  const out = [];
  for (let i = 0; i < n; i += 1) {
    out.push([
      dequantizeFromBins(lVals[i], 0, 1, cfg.NUM_BINS_L),
      dequantizeFromBins(cVals[i], 0, cfg.C_MAX, cfg.NUM_BINS_C),
      dequantizeFromBins(hVals[i], 0, 1, cfg.NUM_BINS_H),
    ]);
  }
  return out;
}

function oklchToSrgb(c) {
  const L = c[0];
  const C = c[1];
  const h = c[2] * Math.PI * 2;

  const a = C * Math.cos(h);
  const b = C * Math.sin(h);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  let rLin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let gLin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let bLin = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;

  rLin = clamp01(rLin);
  gLin = clamp01(gLin);
  bLin = clamp01(bLin);

  const encode = (u) => (u <= 0.0031308 ? 12.92 * u : 1.055 * Math.pow(u, 1 / 2.4) - 0.055);

  return {
    r: clamp01(encode(rLin)),
    g: clamp01(encode(gLin)),
    b: clamp01(encode(bLin)),
    rLin,
    gLin,
    bLin,
  };
}

function relativeLuminance(oklch) {
  const { rLin, gLin, bLin } = oklchToSrgb(oklch);
  return 0.2126 * rLin + 0.7152 * gLin + 0.0722 * bLin;
}

function contrastRatio(c1, c2) {
  const l1 = relativeLuminance(c1);
  const l2 = relativeLuminance(c2);
  const hi = Math.max(l1, l2);
  const lo = Math.min(l1, l2);
  return (hi + 0.05) / (lo + 0.05);
}

function buildMutedColor(text, bg) {
  return [0.65 * text[0] + 0.35 * bg[0], Math.max(0.015, text[1] * 0.35), text[2]];
}

function chooseRoles(palette, cMax, darkMode) {
  if (palette.length < 4) return null;

  const profiles = [
    { name: "strict", textBg: 4.5, textSurface: 4.5, accentBg: 2.8 },
    { name: "relaxed", textBg: 4.5, textSurface: 3.6, accentBg: 1.8 },
  ];

  for (let p = 0; p < profiles.length; p += 1) {
    const profile = profiles[p];
    let best = null;

    for (let bgIdx = 0; bgIdx < 4; bgIdx += 1) {
      for (let surfaceIdx = 0; surfaceIdx < 4; surfaceIdx += 1) {
        if (surfaceIdx === bgIdx) continue;
        for (let textIdx = 0; textIdx < 4; textIdx += 1) {
          if (textIdx === bgIdx || textIdx === surfaceIdx) continue;
          for (let accentIdx = 0; accentIdx < 4; accentIdx += 1) {
            if (accentIdx === bgIdx || accentIdx === surfaceIdx || accentIdx === textIdx) continue;

            const bg = palette[bgIdx];
            const surface = palette[surfaceIdx];
            const text = palette[textIdx];
            const accent = palette[accentIdx];

            if (!(surface[1] + 1e-6 < accent[1])) continue;

            const bgLum = relativeLuminance(bg);
            const textLum = relativeLuminance(text);
            if (darkMode) {
              if (!(bgLum < textLum)) continue;
              if (bgLum > 0.35) continue;
            } else {
              if (!(bgLum > textLum)) continue;
              if (bgLum < 0.25) continue;
            }

            const textContrast = contrastRatio(text, bg);
            if (textContrast < profile.textBg) continue;

            const textSurfaceContrast = contrastRatio(text, surface);
            if (textSurfaceContrast < profile.textSurface) continue;

            const accentContrast = contrastRatio(accent, bg);
            if (accentContrast < profile.accentBg) continue;

            const muted = buildMutedColor(text, bg);
            const mutedContrast = contrastRatio(muted, bg);
            if (mutedContrast < MIN_MUTED_CONTRAST) continue;

            const cNorm = bg[1] / Math.max(cMax, 1e-6);
            const bgSoftness = bg[0] - 0.7 * cNorm;
            const accentVivid = accent[1] / Math.max(cMax, 1e-6);
            const surfaceDistance = Math.abs(surface[0] - bg[0]) + 0.5 * Math.abs(surface[1] - bg[1]);
            const modeBias = darkMode ? (0.35 - bgLum) : (bgLum - 0.25);

            const score =
              1.8 * textContrast +
              0.7 * accentContrast +
              0.6 * textSurfaceContrast +
              0.6 * bgSoftness +
              0.9 * accentVivid -
              0.4 * surfaceDistance +
              1.0 * modeBias;

            if (!best || score > best.score) {
              best = {
                bg,
                surface,
                text,
                accent,
                muted,
                contrast: textContrast,
                textSurfaceContrast,
                accentContrast,
                mutedContrast,
                bgIdx,
                surfaceIdx,
                textIdx,
                accentIdx,
                profile: profile.name,
                score,
              };
            }
          }
        }
      }
    }

    if (best) {
      delete best.score;
      return best;
    }
  }

  return null;
}

function tryAssignRolesByLuminance(palette, darkMode) {
  if (palette.length < 4) return null;

  const indices = [0, 1, 2, 3];
  const lums = indices.map((idx) => ({ idx, lum: relativeLuminance(palette[idx]) }));
  const byLumAsc = lums.slice().sort((a, b) => a.lum - b.lum).map((x) => x.idx);

  const bgOrder = darkMode ? byLumAsc : byLumAsc.slice().reverse();
  const textOrder = darkMode ? byLumAsc.slice().reverse() : byLumAsc;

  let best = null;

  for (let bi = 0; bi < bgOrder.length; bi += 1) {
    const bgIdx = bgOrder[bi];
    const bg = palette[bgIdx];
    const bgLum = relativeLuminance(bg);

    for (let ti = 0; ti < textOrder.length; ti += 1) {
      const textIdx = textOrder[ti];
      if (textIdx === bgIdx) continue;

      const text = palette[textIdx];
      const textLum = relativeLuminance(text);
      const textContrast = contrastRatio(text, bg);
      if (textContrast < 4.2) continue;

      if (darkMode) {
        if (!(bgLum < textLum)) continue;
      } else {
        if (!(bgLum > textLum)) continue;
      }

      const remaining = indices.filter((idx) => idx !== bgIdx && idx !== textIdx);
      if (remaining.length !== 2) continue;

      let accentIdx = remaining[0];
      let surfaceIdx = remaining[1];
      if (palette[remaining[1]][1] > palette[remaining[0]][1]) {
        accentIdx = remaining[1];
        surfaceIdx = remaining[0];
      }

      const accent = palette[accentIdx];
      const surface = palette[surfaceIdx];
      const accentContrast = contrastRatio(accent, bg);
      const textSurfaceContrast = contrastRatio(text, surface);
      if (accentContrast < 1.45) continue;
      if (textSurfaceContrast < 3.0) continue;

      const muted = buildMutedColor(text, bg);
      const mutedContrast = contrastRatio(muted, bg);
      if (mutedContrast < 2.5) continue;

      const score =
        2.0 * textContrast +
        0.8 * textSurfaceContrast +
        0.5 * accentContrast +
        0.35 * (accent[1] - surface[1]) -
        0.2 * Math.abs(surface[0] - bg[0]);

      if (!best || score > best.score) {
        best = {
          bg,
          surface,
          text,
          accent,
          muted,
          contrast: textContrast,
          textSurfaceContrast,
          accentContrast,
          mutedContrast,
          bgIdx,
          surfaceIdx,
          textIdx,
          accentIdx,
          profile: darkMode ? "luminance-dark" : "luminance-light",
          score,
        };
      }
    }
  }

  if (!best) return null;
  delete best.score;
  return best;
}

function chooseRolesByPreference(palette, cMax, darkModePref) {
  const tryDark = () => tryAssignRolesByLuminance(palette, true) || chooseRoles(palette, cMax, true);
  const tryLight = () => tryAssignRolesByLuminance(palette, false) || chooseRoles(palette, cMax, false);

  if (darkModePref === true) return tryDark();
  if (darkModePref === false) return tryLight();

  const dark = tryDark();
  const light = tryLight();
  if (dark && light) {
    return dark.contrast >= light.contrast ? dark : light;
  }
  return dark || light;
}

function chooseRolesClassicByPreference(palette, cMax, darkModePref) {
  if (darkModePref === true) return chooseRoles(palette, cMax, true);
  if (darkModePref === false) return chooseRoles(palette, cMax, false);

  const rolesLight = chooseRoles(palette, cMax, false);
  const rolesDark = chooseRoles(palette, cMax, true);
  if (rolesLight && rolesDark) {
    const bgLum = relativeLuminance(rolesLight.bg);
    return bgLum >= 0.5 ? rolesLight : rolesDark;
  }
  return rolesLight || rolesDark;
}

function remapPastelRolesForDarkMode(palette, fallbackRoles) {
  if (!palette || palette.length < 4) return fallbackRoles;

  const ranked = [0, 1, 2, 3]
    .map((idx) => ({ idx, l: palette[idx][0] }))
    .sort((a, b) => a.l - b.l)
    .map((x) => x.idx);

  const bgIdx = ranked[0];
  const textIdx = ranked[ranked.length - 1];
  const remaining = ranked.filter((idx) => idx !== bgIdx && idx !== textIdx);
  if (remaining.length !== 2) return fallbackRoles;

  let accentIdx = remaining[0];
  let surfaceIdx = remaining[1];
  if (palette[remaining[1]][1] > palette[remaining[0]][1]) {
    accentIdx = remaining[1];
    surfaceIdx = remaining[0];
  }

  const bg = palette[bgIdx];
  const text = palette[textIdx];
  const surface = palette[surfaceIdx];
  const accent = palette[accentIdx];

  const textContrast = contrastRatio(text, bg);
  const textSurfaceContrast = contrastRatio(text, surface);
  const accentContrast = contrastRatio(accent, bg);
  const muted = buildMutedColor(text, bg);
  const mutedContrast = contrastRatio(muted, bg);

  return {
    bg,
    surface,
    text,
    accent,
    muted,
    contrast: textContrast,
    textSurfaceContrast,
    accentContrast,
    mutedContrast,
    bgIdx,
    surfaceIdx,
    textIdx,
    accentIdx,
    profile: "pastel-dark-remap",
  };
}

function randomColorForMode(mode, cMax) {
  if (mode === "grey") {
    const l = 0.34 + Math.random() * 0.5;
    const c = Math.random() * 0.018;
    return [l, Math.min(c, 0.12 * cMax), Math.random()];
  }

  if (mode === "pastel") {
    const l = 0.66 + Math.random() * 0.24;
    const c = 0.02 + Math.random() * 0.045;
    return [l, Math.min(c, 0.55 * cMax), Math.random()];
  }

  if (mode === "computer90s" || mode === "extremeComputer90s") {
    const r = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
    const g = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
    const b = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
    return rgbBytesToOklch(r, g, b, cMax);
  }

  if (mode === "retro") {
    const baseHue = RETRO_HUES[Math.floor(Math.random() * RETRO_HUES.length)];
    const h = (baseHue + (Math.random() - 0.5) * 0.05 + 1) % 1;
    const l = 0.4 + Math.random() * 0.32;
    const c = 0.06 + Math.random() * 0.08;
    return [l, Math.min(c, cMax), h];
  }

  const l = 0.5 + Math.random() * (0.82 - 0.5);
  let cLow = Math.min(Math.max(0.08, 0.55 * cMax), cMax);
  const cHigh = cMax;
  if (cLow >= cHigh) cLow = 0.5 * cHigh;
  const c = cLow + Math.random() * (cHigh - cLow);
  const h = Math.random();
  return [l, c, h];
}

function randomWebSafeOklch(cMax) {
  const r = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
  const g = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
  const b = WEBSAFE_STEPS[Math.floor(Math.random() * WEBSAFE_STEPS.length)];
  return rgbBytesToOklch(r, g, b, cMax);
}

function countVibrantColors(palette, cMax) {
  const cThresh = Math.min(0.12, 0.95 * cMax);
  let n = 0;
  for (let i = 0; i < palette.length; i += 1) {
    const c = palette[i];
    if (c[1] >= cThresh && c[0] >= 0.35 && c[0] <= 0.82) n += 1;
  }
  return n;
}

function isPastelPalette(palette, cMax) {
  const cSoftMax = Math.min(0.1, 0.7 * cMax);
  let softCount = 0;
  let darkAnchorCount = 0;

  for (let i = 0; i < palette.length; i += 1) {
    const c = palette[i];
    if (c[1] > cSoftMax) return false;

    // Keep a soft palette while allowing one darker low-chroma anchor for readable text.
    if (c[0] >= 0.62 && c[0] <= 0.95) softCount += 1;
    else if (c[0] >= 0.18 && c[0] <= 0.5) darkAnchorCount += 1;
    else return false;
  }
  return softCount >= 3 && darkAnchorCount >= 1;
}

function isPastelColor(color, cMax) {
  const cSoftMax = Math.min(0.1, 0.7 * cMax);
  if (color[1] > cSoftMax) return false;
  const l = color[0];
  return (l >= 0.62 && l <= 0.95) || (l >= 0.18 && l <= 0.5);
}

function pastelBand(color, cMax) {
  const cSoftMax = Math.min(0.1, 0.7 * cMax);
  if (color[1] > cSoftMax) return null;
  if (color[0] >= 0.62 && color[0] <= 0.95) return "soft";
  if (color[0] >= 0.18 && color[0] <= 0.5) return "dark";
  return null;
}

function randomPastelPrefixColor(cMax, preferDarkAnchor) {
  if (preferDarkAnchor) {
    const l = 0.22 + Math.random() * 0.26;
    const c = 0.015 + Math.random() * 0.05;
    return [l, Math.min(c, 0.55 * cMax), Math.random()];
  }

  const l = 0.66 + Math.random() * 0.24;
  const c = 0.02 + Math.random() * 0.045;
  return [l, Math.min(c, 0.55 * cMax), Math.random()];
}

function dequantizedColorFromTokens(rt, tokenTriplet) {
  const cfg = rt.cfg;
  return [
    dequantizeFromBins(tokenTriplet[0] - cfg.L_BASE, 0, 1, cfg.NUM_BINS_L),
    dequantizeFromBins(tokenTriplet[1] - cfg.C_BASE, 0, cfg.C_MAX, cfg.NUM_BINS_C),
    dequantizeFromBins(tokenTriplet[2] - cfg.H_BASE, 0, 1, cfg.NUM_BINS_H),
  ];
}

function generateValidatedPastelPrefixColor(rt, cMax, preferDarkAnchor) {
  for (let i = 0; i < 24; i += 1) {
    const candidate = randomPastelPrefixColor(cMax, preferDarkAnchor);
    if (!isPastelColor(candidate, cMax)) continue;

    // Validate after model binning so the actual prefix fed to generation remains pastel.
    const tokens = oklchColorToTokens(rt, candidate);
    const dequantized = dequantizedColorFromTokens(rt, tokens);
    if (isPastelColor(dequantized, cMax)) return dequantized;
  }

  return randomPastelPrefixColor(cMax, preferDarkAnchor);
}

function circularHueDistance(a, b) {
  const d = Math.abs(a - b);
  return Math.min(d, 1 - d);
}

function isRetroPalette(palette) {
  let nearRetro = 0;
  for (let i = 0; i < palette.length; i += 1) {
    const col = palette[i];
    const isNear = RETRO_HUES.some((h) => circularHueDistance(col[2], h) <= 22 / 360);
    if (isNear && col[1] >= 0.05 && col[1] <= 0.16 && col[0] >= 0.28 && col[0] <= 0.8) nearRetro += 1;
  }
  return nearRetro >= 3;
}

function isGreyPalette(palette, cMax) {
  const veryLowChroma = Math.min(0.05, 0.26 * cMax);
  const lowChroma = Math.min(0.075, 0.36 * cMax);
  let nearNeutralCount = 0;

  for (let i = 0; i < palette.length; i += 1) {
    const c = palette[i][1];
    if (c > lowChroma) return false;
    if (c <= veryLowChroma) nearNeutralCount += 1;
  }

  return nearNeutralCount >= 3;
}

function srgbByteToLinear(v) {
  const x = v / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

function rgbBytesToOklch(r, g, b, cMax) {
  const rr = srgbByteToLinear(r);
  const gg = srgbByteToLinear(g);
  const bb = srgbByteToLinear(b);

  const l = 0.4122214708 * rr + 0.5363325363 * gg + 0.0514459929 * bb;
  const m = 0.2119034982 * rr + 0.6806995451 * gg + 0.1073969566 * bb;
  const s = 0.0883024619 * rr + 0.2817188376 * gg + 0.6299787005 * bb;

  const l3 = Math.cbrt(l);
  const m3 = Math.cbrt(m);
  const s3 = Math.cbrt(s);

  const L = 0.2104542553 * l3 + 0.793617785 * m3 - 0.0040720468 * s3;
  const a = 1.9779984951 * l3 - 2.428592205 * m3 + 0.4505937099 * s3;
  const b2 = 0.0259040371 * l3 + 0.7827717662 * m3 - 0.808675766 * s3;

  let h = Math.atan2(b2, a) / (2 * Math.PI);
  if (h < 0) h += 1;

  return [clamp01(L), Math.max(0, Math.min(Math.sqrt(a * a + b2 * b2), cMax)), h];
}

function normalizeForcedColor(color, cMax) {
  if (!color) return null;
  return [clamp01(color[0]), Math.max(0, Math.min(color[1], cMax)), ((color[2] % 1) + 1) % 1];
}

function resolveTemperaturePreset(profile) {
  if (profile && TEMPERATURE_PRESETS[profile]) return TEMPERATURE_PRESETS[profile];
  return TEMPERATURE_PRESETS.medium;
}

function generatePaletteInWorker(rt, options, requestId) {
  const cfg = rt.cfg;
  const mode = MODE_RULES[options.mode] ? options.mode : "normal";
  const rules = MODE_RULES[mode];
  const temps = resolveTemperaturePreset(options.temperatureProfile);
  const darkMode = typeof options.darkMode === "boolean" ? options.darkMode : null;
  const forcedA = normalizeForcedColor(options.forcedA, cfg.C_MAX);
  const forcedB = normalizeForcedColor(options.forcedB, cfg.C_MAX);

  if (rules.pastel && forcedA && !isPastelColor(forcedA, cfg.C_MAX)) {
    return { palette: null, roles: null, attempts: 0, vibrantCount: 0, maxAttempts: rules.maxAttempts };
  }

  if (rules.pastel && forcedB && !isPastelColor(forcedB, cfg.C_MAX)) {
    return { palette: null, roles: null, attempts: 0, vibrantCount: 0, maxAttempts: rules.maxAttempts };
  }

  if (rules.grey && forcedA && forcedA[1] > Math.min(0.075, 0.36 * cfg.C_MAX)) {
    return { palette: null, roles: null, attempts: 0, vibrantCount: 0, maxAttempts: rules.maxAttempts };
  }

  if (rules.grey && forcedB && forcedB[1] > Math.min(0.075, 0.36 * cfg.C_MAX)) {
    return { palette: null, roles: null, attempts: 0, vibrantCount: 0, maxAttempts: rules.maxAttempts };
  }

  if (rules.pastel && forcedA && forcedB) {
    const bandA = pastelBand(forcedA, cfg.C_MAX);
    const bandB = pastelBand(forcedB, cfg.C_MAX);
    if (bandA === "dark" && bandB === "dark") {
      return { palette: null, roles: null, attempts: 0, vibrantCount: 0, maxAttempts: rules.maxAttempts };
    }
  }

  for (let attempt = 1; attempt <= rules.maxAttempts; attempt += 1) {
    if (attempt % 8 === 0) {
      postMessage({ type: "progress", requestId, attempt, maxAttempts: rules.maxAttempts });
    }

    const preferDarkPastelPrefix = rules.pastel && !forcedA && attempt % 3 === 0;
    const first =
      forcedA ||
      (rules.pastel
        ? generateValidatedPastelPrefixColor(rt, cfg.C_MAX, preferDarkPastelPrefix)
        : randomColorForMode(mode, cfg.C_MAX));
    const second =
      forcedB || (rules.extremeComputer90s ? randomWebSafeOklch(cfg.C_MAX) : null);

    const firstTokens = oklchColorToTokens(rt, first);
    const prefix = [cfg.BOS_TOKEN, firstTokens[0], firstTokens[1], firstTokens[2], cfg.SEP_TOKEN];

    if (second) {
      const secondTokens = oklchColorToTokens(rt, second);
      prefix.push(secondTokens[0], secondTokens[1], secondTokens[2], cfg.SEP_TOKEN);
    }

    const seq = sampleFromPrefixFastDynamic(rt, prefix, temps.first, temps.middle, temps.last);
    const decoded = decodePaletteTokens(rt, stripSpecial(rt, seq));
    if (!decoded || decoded.length < 4) continue;

    const palette = decoded.slice(0, 4);
    if (forcedA) palette[0] = forcedA;
    if (second) palette[1] = second;

    const vibrantCount = countVibrantColors(palette, cfg.C_MAX);
    if (vibrantCount < rules.minVibrant) continue;
    if (rules.pastel && !isPastelPalette(palette, cfg.C_MAX)) continue;
    if (rules.retro && !isRetroPalette(palette)) continue;
    if (rules.grey && !isGreyPalette(palette, cfg.C_MAX)) continue;

    let roles;
    if (rules.pastel) {
      // Pastel generation should not be blocked by dark-mode constraints.
      const basePastelRoles = chooseRolesByPreference(palette, cfg.C_MAX, null);
      if (!basePastelRoles) continue;
      roles = darkMode === true ? remapPastelRolesForDarkMode(palette, basePastelRoles) : basePastelRoles;
    } else {
      roles = chooseRolesClassicByPreference(palette, cfg.C_MAX, darkMode);
    }

    if (!roles) continue;

    return { palette, roles, attempts: attempt, vibrantCount, maxAttempts: rules.maxAttempts };
  }

  return { palette: null, roles: null, attempts: rules.maxAttempts, vibrantCount: 0, maxAttempts: rules.maxAttempts };
}

self.onmessage = async (event) => {
  const msg = event.data || {};

  try {
    if (msg.type === "init") {
      let flat;
      if (msg.weightsBase64) flat = loadBinaryWeightsFromBase64(msg.weightsBase64);
      else if (msg.weightsUrl) flat = await loadBinaryWeightsFromUrl(msg.weightsUrl);
      else throw new Error("Missing worker init payload (weightsBase64 or weightsUrl).");
      runtime = buildRuntimeFromFlat(flat);
      postMessage({ type: "ready" });
      return;
    }

    if (msg.type === "generate") {
      if (!runtime) throw new Error("Worker not initialized.");
      const result = generatePaletteInWorker(runtime, msg.options || {}, msg.requestId);
      postMessage({ type: "done", requestId: msg.requestId, result });
    }
  } catch (err) {
    postMessage({ type: "error", requestId: msg.requestId, message: err.message || String(err) });
  }
};
