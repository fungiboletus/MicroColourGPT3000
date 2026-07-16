const C_MAX = 0.21791855751774927;
const SESSION_KEYS = {
  mode: "palette_mode",
  themePref: "palette_theme_pref",
  neutralAnchor: "palette_neutral_anchor",
  temperatureProfile: "palette_temperature_profile",
  lockA: "palette_lock_a",
  lockB: "palette_lock_b",
  pickerSource: "palette_picker_source",
};

const TEMPERATURE_PROFILES = {
  low: { label: "Low" },
  medium: { label: "Medium" },
  high: { label: "High" },
};

const worker = new Worker("./paletteWorker.js?v=a");
let workerReady = false;
let activeRequestId = 0;
let latestRoles = null;
let latestRawPalette = null;
let latestProtectedCount = 0;
let requestedProtectedCount = 0;
let latestResultMeta = null;
const DEFAULT_GENERATE_LABEL = "Generate Palette";

const attemptsEl = document.getElementById("attemptInfo");
const statusRefreshBtn = document.getElementById("statusRefresh");
const btnGenerate = document.getElementById("btnGenerate");
const btnTemperature = document.getElementById("btnTemperature");

const lockAInput = document.getElementById("lockA");
const lockBInput = document.getElementById("lockB");
const previewA = document.getElementById("previewA");
const previewB = document.getElementById("previewB");
const validA = document.getElementById("validA");
const validB = document.getElementById("validB");
const paletteEl = document.getElementById("palette");
const appRoot = document.getElementById("appRoot");
const materialPicker = document.getElementById("materialPicker");
const pickerClearBtn = document.getElementById("pickerClear");
const pickerCloseBtn = document.getElementById("pickerClose");
const pickerGrid = document.getElementById("pickerGrid");
const pickerSourceMdBtn = document.getElementById("pickerSourceMd");
const pickerSourceTwBtn = document.getElementById("pickerSourceTw");
const temperatureDialog = document.getElementById("temperatureDialog");
const temperatureOptions = document.getElementById("temperatureOptions");
const temperatureCloseBtn = document.getElementById("temperatureClose");

let selectedTemperatureProfile = "medium";

const parseCanvas = document.createElement("canvas");
parseCanvas.width = 1;
parseCanvas.height = 1;
const parseCtx = parseCanvas.getContext("2d", { willReadFrequently: true });
const PICKER_DATA = window.PICKER_COLOR_DATA || {};
let materialGroups = Array.isArray(PICKER_DATA.material) ? PICKER_DATA.material : [];
let tailwindGroups = Array.isArray(PICKER_DATA.tailwind) ? PICKER_DATA.tailwind : [];
let activePickerSlot = null;
let activePickerSource = "md";

function setStatus(message, busy, isError = false, canRefresh = false) {
  btnGenerate.disabled = !!busy || !workerReady;
  btnGenerate.classList.toggle("isBusy", !!busy);
  btnGenerate.textContent = busy ? "Generating..." : DEFAULT_GENERATE_LABEL;

  attemptsEl.textContent = message || "";
  attemptsEl.classList.toggle("error", !!isError);

  if (statusRefreshBtn) {
    const showRefresh = !!canRefresh && !!isError;
    statusRefreshBtn.hidden = !showRefresh;
    statusRefreshBtn.setAttribute("aria-hidden", String(!showRefresh));
  }
}

function selectedMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "normal";
}

function selectedThemePref() {
  const checked = document.querySelector('input[name="themePref"]:checked');
  return checked ? checked.value : "auto";
}

function selectedNeutralAnchor() {
  const checked = document.querySelector('input[name="neutralAnchor"]:checked');
  return checked ? checked.value : "auto";
}

function saveSessionState() {
  try {
    sessionStorage.setItem(SESSION_KEYS.mode, selectedMode());
    sessionStorage.setItem(SESSION_KEYS.themePref, selectedThemePref());
    sessionStorage.setItem(SESSION_KEYS.neutralAnchor, selectedNeutralAnchor());
    sessionStorage.setItem(SESSION_KEYS.temperatureProfile, selectedTemperatureProfile);
    sessionStorage.setItem(SESSION_KEYS.lockA, lockAInput.value.trim());
    sessionStorage.setItem(SESSION_KEYS.lockB, lockBInput.value.trim());
    sessionStorage.setItem(SESSION_KEYS.pickerSource, activePickerSource);
  } catch {
    // Ignore storage errors (private mode, disabled storage, etc.).
  }
}

function restoreSessionState() {
  try {
    const savedMode = sessionStorage.getItem(SESSION_KEYS.mode);
    const savedThemePref = sessionStorage.getItem(SESSION_KEYS.themePref);
    const savedNeutralAnchor = sessionStorage.getItem(SESSION_KEYS.neutralAnchor);
    const savedTemperatureProfile = sessionStorage.getItem(SESSION_KEYS.temperatureProfile);
    const savedLockA = sessionStorage.getItem(SESSION_KEYS.lockA);
    const savedLockB = sessionStorage.getItem(SESSION_KEYS.lockB);
    const savedPickerSource = sessionStorage.getItem(SESSION_KEYS.pickerSource);

    if (savedMode) {
      const modeInput = document.querySelector(`input[name="mode"][value="${savedMode}"]`);
      if (modeInput) modeInput.checked = true;
    }

    if (savedThemePref) {
      const themeInput = document.querySelector(`input[name="themePref"][value="${savedThemePref}"]`);
      if (themeInput) themeInput.checked = true;
    }

    if (savedNeutralAnchor) {
      const anchorInput = document.querySelector(`input[name="neutralAnchor"][value="${savedNeutralAnchor}"]`);
      if (anchorInput) anchorInput.checked = true;
    }

    if (savedTemperatureProfile && TEMPERATURE_PROFILES[savedTemperatureProfile]) {
      selectedTemperatureProfile = savedTemperatureProfile;
    }

    if (savedLockA !== null) lockAInput.value = savedLockA;
    if (savedLockB !== null) lockBInput.value = savedLockB;
    if (savedPickerSource === "tw" || savedPickerSource === "md") {
      activePickerSource = savedPickerSource;
    }
  } catch {
    // Ignore storage errors (private mode, disabled storage, etc.).
  }
}

function applyTemperatureProfileUi() {
  const profile = TEMPERATURE_PROFILES[selectedTemperatureProfile] ? selectedTemperatureProfile : "medium";
  selectedTemperatureProfile = profile;

  const input = document.querySelector(`input[name="temperatureProfile"][value="${profile}"]`);
  if (input) input.checked = true;

  if (btnTemperature) {
    const label = TEMPERATURE_PROFILES[profile].label;
    btnTemperature.setAttribute("title", `Temperature: ${label}`);
    btnTemperature.setAttribute("aria-label", `Set generation temperature (current: ${label})`);
  }
}

function openTemperatureDialog() {
  if (!temperatureDialog) return;
  applyTemperatureProfileUi();
  if (temperatureDialog.open) return;

  if (typeof temperatureDialog.showModal === "function") {
    try {
      temperatureDialog.showModal();
      requestAnimationFrame(() => {
        if (btnTemperature) positionTemperatureDialogForButton(btnTemperature);
      });
      return;
    } catch {
      // Fall through to open attribute fallback.
    }
  }

  temperatureDialog.setAttribute("open", "");
  requestAnimationFrame(() => {
    if (btnTemperature) positionTemperatureDialogForButton(btnTemperature);
  });
}

function closeTemperatureDialog() {
  if (!temperatureDialog || !temperatureDialog.open) return;
  if (typeof temperatureDialog.close === "function") {
    temperatureDialog.close();
    return;
  }
  temperatureDialog.removeAttribute("open");
}

function colorToRgbObject(cssColor) {
  const raw = cssColor.trim();
  if (!raw || !parseCtx) return null;

  if (typeof CSS !== "undefined" && typeof CSS.supports === "function" && !CSS.supports("color", raw)) {
    return null;
  }

  parseCtx.clearRect(0, 0, 1, 1);
  parseCtx.fillStyle = raw;
  parseCtx.fillRect(0, 0, 1, 1);

  const [r, g, b, a] = parseCtx.getImageData(0, 0, 1, 1).data;
  return { r, g, b, alpha: a / 255 };
}

function cssColorToRgb(input) {
  const parsed = colorToRgbObject(input);
  if (!parsed) return null;
  return { r: parsed.r, g: parsed.g, b: parsed.b };
}

function srgbByteToLinear(v) {
  const x = v / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

function rgbToOklch(rgb) {
  const r = srgbByteToLinear(rgb.r);
  const g = srgbByteToLinear(rgb.g);
  const b = srgbByteToLinear(rgb.b);

  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l3 = Math.cbrt(l);
  const m3 = Math.cbrt(m);
  const s3 = Math.cbrt(s);

  const L = 0.2104542553 * l3 + 0.793617785 * m3 - 0.0040720468 * s3;
  const a = 1.9779984951 * l3 - 2.428592205 * m3 + 0.4505937099 * s3;
  const b2 = 0.0259040371 * l3 + 0.7827717662 * m3 - 0.808675766 * s3;

  let h = Math.atan2(b2, a) / (2 * Math.PI);
  if (h < 0) h += 1;

  return [Math.max(0, Math.min(1, L)), Math.sqrt(a * a + b2 * b2), h];
}

function oklchToCss(c) {
  return `oklch(${(c[0] * 100).toFixed(1)}% ${c[1].toFixed(3)} ${(c[2] * 360).toFixed(1)}deg)`;
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

  rLin = Math.max(0, Math.min(1, rLin));
  gLin = Math.max(0, Math.min(1, gLin));
  bLin = Math.max(0, Math.min(1, bLin));

  const encode = (u) => (u <= 0.0031308 ? 12.92 * u : 1.055 * Math.pow(u, 1 / 2.4) - 0.055);

  return {
    r: Math.max(0, Math.min(255, Math.round(encode(rLin) * 255))),
    g: Math.max(0, Math.min(255, Math.round(encode(gLin) * 255))),
    b: Math.max(0, Math.min(255, Math.round(encode(bLin) * 255))),
  };
}

function rgbToHex(rgb) {
  const c = (n) => n.toString(16).padStart(2, "0");
  return `#${c(rgb.r)}${c(rgb.g)}${c(rgb.b)}`;
}

function relativeLuminance(rgb) {
  const toLinear = (v) => {
    const x = v / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  };

  const r = toLinear(rgb.r);
  const g = toLinear(rgb.g);
  const b = toLinear(rgb.b);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(l1, l2) {
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

function colorContrast(a, b) {
  return contrastRatio(relativeLuminance(a), relativeLuminance(b));
}

function toCandidate(oklchColor) {
  const rgb = oklchToSrgb(oklchColor);
  return { css: oklchToCss(oklchColor), rgb };
}

function chooseAccessibleColor(candidates, background, minRatio) {
  const bgRgb = oklchToSrgb(background);
  let best = null;

  for (const cand of candidates) {
    const ratio = colorContrast(cand.rgb, bgRgb);
    if (ratio >= minRatio) return cand;
    if (!best || ratio > best.ratio) best = { cand, ratio };
  }

  const fallback = [
    { css: "#000000", rgb: { r: 0, g: 0, b: 0 } },
    { css: "#ffffff", rgb: { r: 255, g: 255, b: 255 } },
  ];

  for (const cand of fallback) {
    const ratio = colorContrast(cand.rgb, bgRgb);
    if (!best || ratio > best.ratio) best = { cand, ratio };
  }

  return best.cand;
}

function validateLockedColor(inputEl, previewEl, msgEl) {
  const raw = inputEl.value.trim();
  if (!raw) {
    previewEl.style.background = "";
    previewEl.classList.remove("hasColor");
    previewEl.classList.add("isPickerIcon");
    previewEl.classList.remove("isLoading");
    previewEl.textContent = "";
    msgEl.textContent = "";
    msgEl.classList.remove("error");
    return null;
  }

  const rgb = cssColorToRgb(raw);
  if (!rgb) {
    previewEl.style.background = "";
    previewEl.classList.remove("hasColor");
    previewEl.classList.add("isPickerIcon");
    previewEl.classList.remove("isLoading");
    previewEl.textContent = "";
    msgEl.textContent = "Invalid CSS colour";
    msgEl.classList.add("error");
    return undefined;
  }

  previewEl.style.background = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  previewEl.classList.add("hasColor");
  previewEl.classList.remove("isPickerIcon");
  previewEl.classList.remove("isLoading");
  previewEl.textContent = "";
  msgEl.textContent = "";
  msgEl.classList.remove("error");

  const oklch = rgbToOklch(rgb);
  oklch[1] = Math.max(0, Math.min(C_MAX, oklch[1]));
  return oklch;
}

function applyTheme(roles) {
  const roleCandidates = [roles.text, roles.muted, roles.accent, roles.bg, roles.surface].map(toCandidate);
  const inkSurface = chooseAccessibleColor(roleCandidates, roles.surface, 4.5);
  const inkBg = chooseAccessibleColor(roleCandidates, roles.bg, 4.5);
  const inkAccent = chooseAccessibleColor(roleCandidates, roles.accent, 4.5);
  const inkControls = chooseAccessibleColor(
    [toCandidate(roles.text), toCandidate(roles.muted), toCandidate(roles.surface), toCandidate(roles.accent), toCandidate(roles.bg)],
    roles.bg,
    4.5,
  );
  const titleInk = chooseAccessibleColor(
    [toCandidate(roles.text), toCandidate(roles.accent), toCandidate(roles.muted), toCandidate(roles.bg)],
    roles.surface,
    4.5,
  );
  const hintInk = chooseAccessibleColor([toCandidate(roles.muted), toCandidate(roles.text), toCandidate(roles.bg)], roles.surface, 4.5);

  appRoot.style.setProperty("--bg", oklchToCss(roles.bg));
  appRoot.style.setProperty("--surface", oklchToCss(roles.surface));
  appRoot.style.setProperty("--text", oklchToCss(roles.text));
  appRoot.style.setProperty("--muted", oklchToCss(roles.muted));
  appRoot.style.setProperty("--accent", oklchToCss(roles.accent));
  appRoot.style.setProperty("--ink-surface", inkSurface.css);
  appRoot.style.setProperty("--ink-bg", inkBg.css);
  appRoot.style.setProperty("--ink-accent", inkAccent.css);
  appRoot.style.setProperty("--ink-controls", inkControls.css);
  appRoot.style.setProperty("--title-ink", titleInk.css);
  appRoot.style.setProperty("--hint-ink", hintInk.css);
  appRoot.style.setProperty("--ink-surface-rgb", `${inkSurface.rgb.r} ${inkSurface.rgb.g} ${inkSurface.rgb.b}`);
  appRoot.style.setProperty("--ink-controls-rgb", `${inkControls.rgb.r} ${inkControls.rgb.g} ${inkControls.rgb.b}`);

  document.documentElement.style.setProperty("--bg", oklchToCss(roles.bg));
  document.documentElement.style.setProperty("--surface", oklchToCss(roles.surface));
  document.documentElement.style.setProperty("--text", oklchToCss(roles.text));
  document.documentElement.style.setProperty("--muted", oklchToCss(roles.muted));
  document.documentElement.style.setProperty("--accent", oklchToCss(roles.accent));
  document.documentElement.style.setProperty("--ink-surface", inkSurface.css);
  document.documentElement.style.setProperty("--ink-bg", inkBg.css);
  document.documentElement.style.setProperty("--ink-accent", inkAccent.css);
  document.documentElement.style.setProperty("--ink-controls", inkControls.css);
  document.documentElement.style.setProperty("--title-ink", titleInk.css);
  document.documentElement.style.setProperty("--hint-ink", hintInk.css);
  document.documentElement.style.setProperty("--ink-surface-rgb", `${inkSurface.rgb.r} ${inkSurface.rgb.g} ${inkSurface.rgb.b}`);
  document.documentElement.style.setProperty("--ink-controls-rgb", `${inkControls.rgb.r} ${inkControls.rgb.g} ${inkControls.rgb.b}`);
}

function clearLockedColor(slot) {
  if (slot === "A") lockAInput.value = "";
  else lockBInput.value = "";
  refreshLockedColorValidation();
  saveSessionState();
  generate();
}

function pickerSourceLabel(source) {
  return source === "tw" ? "Tailwind" : "Material";
}

function activePickerGroups() {
  return activePickerSource === "tw" ? tailwindGroups : materialGroups;
}

function updatePickerSourceButtons() {
  if (!pickerSourceMdBtn || !pickerSourceTwBtn) return;
  const mdActive = activePickerSource === "md";
  pickerSourceMdBtn.classList.toggle("isActive", mdActive);
  pickerSourceTwBtn.classList.toggle("isActive", !mdActive);
  pickerSourceMdBtn.setAttribute("aria-pressed", String(mdActive));
  pickerSourceTwBtn.setAttribute("aria-pressed", String(!mdActive));
}

function setPickerSource(source) {
  activePickerSource = source === "tw" ? "tw" : "md";
  updatePickerSourceButtons();
  saveSessionState();
  renderPickerGrid();

  if (materialPicker.open && activePickerSlot) {
    requestAnimationFrame(() => {
      const buttonEl = activePickerSlot === "A" ? previewA : previewB;
      positionPickerForButton(buttonEl);
    });
  }
}

function renderPickerGrid() {
  pickerGrid.innerHTML = "";
  pickerGrid.style.width = "";
  materialPicker.style.width = "";

  const groups = activePickerGroups();
  if (!groups.length) {
    const fallback = document.createElement("p");
    fallback.className = "pickerLoading";
    fallback.textContent = `Loading ${pickerSourceLabel(activePickerSource)} swatches...`;
    pickerGrid.appendChild(fallback);
    return;
  }

  const swatchSize = 24;
  const useBasicsPins = activePickerSource === "md";
  const normalGroups = useBasicsPins ? groups.filter((group) => group.name !== "Basics") : groups;
  const basicsGroup = useBasicsPins ? groups.find((group) => group.name === "Basics") || null : null;

  for (const group of normalGroups) {
    const col = document.createElement("section");
    col.className = "pickerCategoryCol";
    col.setAttribute("role", "listitem");
    col.setAttribute("aria-label", `${group.name} swatches`);

    for (const color of group.colors) {
      const swatch = document.createElement("button");
      swatch.type = "button";
      swatch.className = "pickerSwatch";
      const cssValue = typeof color === "string" ? color : color.css || color.hex;
      const shadeLabel = typeof color === "string" ? "" : ` ${color.shade || ""}`;
      swatch.style.background = cssValue;
      swatch.dataset.color = cssValue;
      swatch.setAttribute("aria-label", `${group.name}${shadeLabel} ${cssValue}`.trim());
      col.appendChild(swatch);
    }

    pickerGrid.appendChild(col);
  }

  if (basicsGroup) {
    const colorValues = basicsGroup.colors.map((color) =>
      typeof color === "string" ? color : color.css || color.hex,
    );
    const white = colorValues.find((v) => v.toLowerCase() === "#ffffff") || null;
    const black = colorValues.find((v) => v.toLowerCase() === "#000000") || null;

    if (white) {
      const swatchWhite = document.createElement("button");
      swatchWhite.type = "button";
      swatchWhite.className = "pickerSwatch pickerSwatchPinned pickerSwatchPinnedWhite";
      swatchWhite.style.background = white;
      swatchWhite.dataset.color = white;
      swatchWhite.setAttribute("aria-label", `Basics White ${white}`);
      pickerGrid.appendChild(swatchWhite);
    }

    if (black) {
      const swatchBlack = document.createElement("button");
      swatchBlack.type = "button";
      swatchBlack.className = "pickerSwatch pickerSwatchPinned pickerSwatchPinnedBlack";
      swatchBlack.style.background = black;
      swatchBlack.dataset.color = black;
      swatchBlack.setAttribute("aria-label", `Basics Black ${black}`);
      pickerGrid.appendChild(swatchBlack);
    }
  }

  const columnCount = Math.max(1, normalGroups.length);
  const contentWidth = columnCount * swatchSize;
  const minDialogWidth = 236;
  const viewportCap = Math.max(minDialogWidth, window.innerWidth - 20);
  const dialogWidth = Math.max(minDialogWidth, Math.min(contentWidth + 6, viewportCap));
  const gridVisibleWidth = Math.max(minDialogWidth - 6, dialogWidth - 6);
  pickerGrid.style.width = `${gridVisibleWidth}px`;
  pickerGrid.style.overflowX = contentWidth > gridVisibleWidth ? "auto" : "hidden";
  materialPicker.style.width = `${dialogWidth}px`;
}

function slotLabel(slot) {
  return slot === "A" ? "Fixed Colour A" : "Fixed Colour B";
}

function positionPickerForButton(buttonEl) {
  const rect = buttonEl.getBoundingClientRect();
  const dialogRect = materialPicker.getBoundingClientRect();
  const margin = 10;

  let top = rect.top - dialogRect.height - margin;
  if (top < margin) top = margin;

  let left = rect.left + rect.width / 2 - dialogRect.width / 2;
  const maxLeft = window.innerWidth - dialogRect.width - margin;
  if (left < margin) left = margin;
  if (left > maxLeft) left = maxLeft;

  materialPicker.style.top = `${Math.round(top)}px`;
  materialPicker.style.left = `${Math.round(left)}px`;
}

function positionTemperatureDialogForButton(buttonEl) {
  if (!temperatureDialog) return;

  const rect = buttonEl.getBoundingClientRect();
  const dialogRect = temperatureDialog.getBoundingClientRect();
  const margin = -35;

  let top = rect.bottom + margin;
  if (top + dialogRect.height > window.innerHeight - margin) {
    top = rect.top - dialogRect.height - margin;
  }
  if (top < margin) top = margin;

  let left = rect.right - dialogRect.width + 4;
  const maxLeft = window.innerWidth - dialogRect.width - margin;
  if (left < margin) left = margin;
  if (left > maxLeft) left = maxLeft;

  temperatureDialog.style.top = `${Math.round(top)}px`;
  temperatureDialog.style.left = `${Math.round(left)}px`;
}

function openMaterialPicker(slot) {
  if (!materialPicker) return;

  activePickerSlot = slot;
  materialPicker.setAttribute("aria-label", `Pick colour for ${slotLabel(slot)}`);
  const activeInput = slot === "A" ? lockAInput : lockBInput;
  pickerClearBtn.disabled = activeInput.value.trim() === "";

  if (!activePickerGroups().length) {
    renderPickerGrid();
  }

  if (!materialPicker.open) {
    materialPicker.showModal();
  }
  requestAnimationFrame(() => {
    const buttonEl = slot === "A" ? previewA : previewB;
    positionPickerForButton(buttonEl);
  });
}

function setLockedColor(slot, color) {
  const rgb = oklchToSrgb(color);
  const hex = rgbToHex(rgb);
  if (slot === "A") lockAInput.value = hex;
  else lockBInput.value = hex;
  refreshLockedColorValidation();
  saveSessionState();
}

function swatchCard(color, index) {
  const rgb = oklchToSrgb(color);
  const hex = rgbToHex(rgb);

  const card = document.createElement("article");
  card.className = "swatch";

  const colorDiv = document.createElement("div");
  colorDiv.className = "swatchColor";
  colorDiv.style.background = oklchToCss(color);
  card.appendChild(colorDiv);

  const meta = document.createElement("div");
  meta.className = "swatchMeta";

  const title = document.createElement("strong");
  title.textContent = `Colour ${index + 1}`;
  meta.appendChild(title);

  const hexRow = document.createElement("div");
  hexRow.className = "hexRow";

  const hexInput = document.createElement("input");
  hexInput.type = "text";
  hexInput.className = "hexInput";
  hexInput.value = hex;
  hexInput.readOnly = true;
  hexInput.setAttribute("aria-label", `Hex value for colour ${index + 1}`);

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "copyHexBtn";
  copyBtn.textContent = "Copy";
  copyBtn.title = "Copy hex colour";
  copyBtn.setAttribute("aria-label", "Copy hex colour");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(hex);
      copyBtn.textContent = "Done";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 900);
    } catch {
      copyBtn.textContent = "Nope";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 900);
    }
  });

  hexRow.appendChild(hexInput);
  hexRow.appendChild(copyBtn);
  meta.appendChild(hexRow);

  const actions = document.createElement("div");
  actions.className = "pinActions";

  const btnA = document.createElement("button");
  btnA.type = "button";
  btnA.textContent = "Use as A";
  btnA.addEventListener("click", () => setLockedColor("A", color));

  const btnB = document.createElement("button");
  btnB.type = "button";
  btnB.textContent = "Use as B";
  btnB.addEventListener("click", () => setLockedColor("B", color));

  actions.appendChild(btnA);
  actions.appendChild(btnB);
  meta.appendChild(actions);

  card.appendChild(meta);
  return card;
}

function renderPalette(palette) {
  paletteEl.innerHTML = "";
  for (let i = 0; i < palette.length; i += 1) {
    paletteEl.appendChild(swatchCard(palette[i], i));
  }
}

function refreshLockedColorValidation() {
  const colorA = validateLockedColor(lockAInput, previewA, validA);
  const colorB = validateLockedColor(lockBInput, previewB, validB);
  return { colorA, colorB };
}

function rolesForPalette(palette, sourceRoles) {
  const bg = palette[sourceRoles.bgIdx];
  const surface = palette[sourceRoles.surfaceIdx];
  const text = palette[sourceRoles.textIdx];
  const accent = palette[sourceRoles.accentIdx];
  return {
    ...sourceRoles,
    bg,
    surface,
    text,
    accent,
    muted: [0.65 * text[0] + 0.35 * bg[0], Math.max(0.015, text[1] * 0.35), text[2]],
    contrast: colorContrast(oklchToSrgb(text), oklchToSrgb(bg)),
  };
}

function paletteWithNeutral(rawPalette, neutral, protectedCount) {
  const palette = rawPalette.map((color) => color.slice());
  const replacingWithWhite = neutral === "white";
  let bestIdx = protectedCount;
  let bestScore = Infinity;

  for (let i = protectedCount; i < palette.length; i += 1) {
    const color = palette[i];
    const targetDistance = replacingWithWhite ? 1 - color[0] : color[0];
    const score = targetDistance * 2 + color[1] * 4;
    if (score < bestScore) {
      bestIdx = i;
      bestScore = score;
    }
  }

  palette[bestIdx] = replacingWithWhite ? [1, 0, 0] : [0, 0, 0];
  return { palette, neutral, neutralIdx: bestIdx };
}

function rolesWithNeutralBackground(palette, neutralIdx, sourceRoles) {
  const remaining = palette.map((_, idx) => idx).filter((idx) => idx !== neutralIdx);
  const textIdx = remaining.reduce((bestIdx, idx) => {
    const bestContrast = colorContrast(oklchToSrgb(palette[bestIdx]), oklchToSrgb(palette[neutralIdx]));
    const contrast = colorContrast(oklchToSrgb(palette[idx]), oklchToSrgb(palette[neutralIdx]));
    return contrast > bestContrast ? idx : bestIdx;
  }, remaining[0]);
  const others = remaining.filter((idx) => idx !== textIdx);
  const accentIdx = palette[others[0]][1] >= palette[others[1]][1] ? others[0] : others[1];
  const surfaceIdx = others.find((idx) => idx !== accentIdx);

  return rolesForPalette(palette, {
    ...sourceRoles,
    bgIdx: neutralIdx,
    textIdx,
    accentIdx,
    surfaceIdx,
  });
}

function transformCurrentPalette() {
  if (!latestRawPalette || !latestResultMeta) return;

  const neutralPref = selectedNeutralAnchor();
  let candidates;
  if (neutralPref === "off") candidates = [{ palette: latestRawPalette.map((color) => color.slice()), neutral: null, neutralIdx: null }];
  else if (neutralPref === "auto") {
    const themePref = selectedThemePref();
    const neutrals = themePref === "dark" ? ["black"] : themePref === "light" ? ["white"] : ["black", "white"];
    candidates = neutrals.map((neutral) => paletteWithNeutral(latestRawPalette, neutral, latestProtectedCount));
  } else {
    candidates = [paletteWithNeutral(latestRawPalette, neutralPref, latestProtectedCount)];
  }

  let best = null;
  const themePref = selectedThemePref();
  for (const candidate of candidates) {
    const neutralMatchesTheme = themePref === "auto"
      || (themePref === "dark" && candidate.neutral === "black")
      || (themePref === "light" && candidate.neutral === "white");
    const roles = candidate.neutralIdx !== null && neutralMatchesTheme
      ? rolesWithNeutralBackground(candidate.palette, candidate.neutralIdx, latestResultMeta.roles)
      : rolesForPalette(candidate.palette, latestResultMeta.roles);
    if (!best || roles.contrast > best.roles.contrast) best = { palette: candidate.palette, roles };
  }

  latestRoles = best.roles;
  applyTheme(best.roles);
  renderPalette(best.palette);
}

function generate() {
  if (!workerReady) return;

  const { colorA, colorB } = refreshLockedColorValidation();
  if (colorA === undefined || colorB === undefined) {
    setStatus("Fix locked colour inputs before generating.", false, true);
    return;
  }

  activeRequestId += 1;
  const requestId = activeRequestId;
  requestedProtectedCount = colorB ? 2 : colorA ? 1 : 0;

  attemptsEl.textContent = "Scanning palette space...";
  setStatus("Scanning palette space...", true);

  const themePref = selectedThemePref();

  worker.postMessage({
    type: "generate",
    requestId,
    options: {
      mode: selectedMode(),
      darkMode: themePref === "dark" ? true : themePref === "light" ? false : null,
      forcedA: colorA || null,
      forcedB: colorB || null,
      temperatureProfile: selectedTemperatureProfile,
    },
  });
}

worker.onmessage = (event) => {
  const msg = event.data || {};

  if (msg.type === "ready") {
    workerReady = true;
    setStatus("", false);
    generate();
    return;
  }

  if (msg.type === "progress") {
    if (msg.requestId !== activeRequestId) return;
    setStatus(`Scan pass ${msg.attempt}...`, true);
    return;
  }

  if (msg.type === "done") {
    if (msg.requestId !== activeRequestId) return;

    const result = msg.result;
    if (!result || !result.roles) {
      setStatus("No valid palette found in this run. Try again.", false, true);
      return;
    }

    latestRoles = result.roles;
    latestRawPalette = result.palette.map((color) => color.slice());
    latestProtectedCount = requestedProtectedCount;
    latestResultMeta = result;

    transformCurrentPalette();
    setStatus(`${result.attempts} attempts - ${result.vibrantCount} vibrant tones`, false);
    return;
  }

  if (msg.type === "error") {
    if (msg.requestId && msg.requestId !== activeRequestId) return;
    const initFailed = !workerReady;
    setStatus(`Worker error: ${msg.message}`, false, true, initFailed);
  }
};

btnGenerate.addEventListener("click", generate);
lockAInput.addEventListener("input", () => {
  refreshLockedColorValidation();
  saveSessionState();
});
lockBInput.addEventListener("input", () => {
  refreshLockedColorValidation();
  saveSessionState();
});
materialPicker.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target === materialPicker) materialPicker.close();
});

materialPicker.addEventListener("close", () => {
  activePickerSlot = null;
});

pickerGrid.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  const swatch = target.closest(".pickerSwatch");
  if (!swatch || !activePickerSlot) return;

  const colorValue = swatch.dataset.color;
  if (!colorValue) return;

  if (activePickerSlot === "A") lockAInput.value = colorValue;
  else lockBInput.value = colorValue;

  refreshLockedColorValidation();
  saveSessionState();
  materialPicker.close();
  generate();
});

pickerClearBtn.addEventListener("click", () => {
  if (!activePickerSlot) return;
  clearLockedColor(activePickerSlot);
  materialPicker.close();
});

pickerCloseBtn.addEventListener("click", () => {
  materialPicker.close();
});

pickerSourceMdBtn.addEventListener("click", () => {
  setPickerSource("md");
});

pickerSourceTwBtn.addEventListener("click", () => {
  setPickerSource("tw");
});

if (btnTemperature) {
  btnTemperature.addEventListener("click", () => {
    openTemperatureDialog();
  });
}

if (temperatureDialog) {
  temperatureDialog.addEventListener("click", (event) => {
    const target = event.target;
    if (target === temperatureDialog) closeTemperatureDialog();
  });
}

if (temperatureCloseBtn) {
  temperatureCloseBtn.addEventListener("click", () => {
    closeTemperatureDialog();
  });
}

if (temperatureOptions) {
  temperatureOptions.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (target.name !== "temperatureProfile") return;
    if (!TEMPERATURE_PROFILES[target.value]) return;

    selectedTemperatureProfile = target.value;
    applyTemperatureProfileUi();
    saveSessionState();

    if (workerReady) generate();
  });
}

if (statusRefreshBtn) {
  statusRefreshBtn.addEventListener("click", () => {
    window.location.reload();
  });
}

window.addEventListener("resize", () => {
  renderPickerGrid();
  if (materialPicker.open && activePickerSlot) {
    const buttonEl = activePickerSlot === "A" ? previewA : previewB;
    positionPickerForButton(buttonEl);
  }
  if (temperatureDialog && temperatureDialog.open && btnTemperature) {
    positionTemperatureDialogForButton(btnTemperature);
  }
});

previewA.addEventListener("click", () => openMaterialPicker("A"));
previewB.addEventListener("click", () => openMaterialPicker("B"));

document.getElementById("modeGroup").addEventListener("change", () => {
  saveSessionState();
  if (latestRoles) generate();
});

document.getElementById("themeGroup").addEventListener("change", () => {
  saveSessionState();
  if (latestRawPalette) transformCurrentPalette();
});

document.getElementById("neutralAnchorGroup").addEventListener("change", () => {
  saveSessionState();
  if (latestRawPalette) transformCurrentPalette();
});

restoreSessionState();
applyTemperatureProfileUi();
refreshLockedColorValidation();
updatePickerSourceButtons();
renderPickerGrid();
setStatus("Loading model in worker...", true);
worker.postMessage({ type: "init", weightsUrl: "./weights/model_weights.f16.bin?v=a" });
