// =========================
// Config (viene del HTML)
// =========================
const CFG = window.LINE_START_CFG || {};
const URLS = CFG.urls || {};

function _cfgStr(x){ return (x == null) ? "" : String(x); }
function _cfgUpper(x){ return _cfgStr(x).toUpperCase(); }

const __IS_APP_LINE = (_cfgStr(CFG.line_key) === "aplicacion");
const __IS_UPLOAD_STAGE = (_cfgStr(CFG.stage) === "upload");

let __currentSubcat = "";  // "aplicacion" | "union"
const __savedMachineBySubcat = { aplicacion: "", union: "" };
const __IS_CICLO_LINE = (_cfgStr(CFG.line_key) === "ciclo_aplicadores");



// ✅ Habilitados por subcategoría
// - Aplicación: Máquina 1 y 2
// - Unión: Máquina 1 y 2  ✅
const APP_ENABLED = {
  aplicacion: ["1","2"],
  union: ["1","2"]
};

// =========================
// ✅ Helpers: usar machine REAL (hidden) y no CFG.machine
// =========================
function getCurrentMachine(){
  return _cfgStr(document.getElementById("machineInput")?.value || CFG.machine);
}
function getCurrentMachineU(){
  return getCurrentMachine().toUpperCase();
}

// =========================
// SUBCATEGORÍA (Aplicación / Unión)
// =========================
function selectSubcat(value, el){
  value = String(value || "").trim().toLowerCase();

  // ✅ Permitir solo subcats habilitadas
  if(!APP_ENABLED[value]){
    alert("Subcategoría no habilitada.");
    return;
  }

  document.querySelectorAll(".subcat-card").forEach(c => c.classList.remove("selected"));
  if(el) el.classList.add("selected");

  __currentSubcat = value;

  const label = document.getElementById("subcatLabel");
  const label2 = document.getElementById("subcatLabel2");
  const in1 = document.getElementById("subcatInput");
  const in2 = document.getElementById("subcatInput2");

  const txt = (value === "union") ? "Unión" : "Aplicación";
  if(label) label.textContent = txt;
  if(label2) label2.textContent = txt;
  if(in1) in1.value = value;
  if(in2) in2.value = value;

  // ✅ Machine real según subcat (para backend)
  const machineInput = document.getElementById("machineInput");
  const machineLabel = document.getElementById("machineLabel");
  const mName = (value === "union") ? "UNION" : "APLICACION";
  if(machineInput) machineInput.value = mName;
  if(machineLabel) machineLabel.textContent = mName;

  // Mostrar dropdown
  const wrap = document.getElementById("appMachineWrap");
  if(wrap) wrap.style.display = "block";

  // ✅ Habilitar/Deshabilitar máquinas según subcat
  const sel = document.getElementById("appMachineSelect");
  const allowed = APP_ENABLED[value] || [];

  if(sel){
    Array.from(sel.options).forEach(op=>{
      const v = String(op.value || "").trim();
      if(!v) return;

      const ok = allowed.includes(v);
      op.disabled = !ok;

      // Manejo del texto "(Próximamente)"
      const base = op.textContent.replace(/\s*\(Próximamente\)\s*/g, "").trim();
      op.textContent = ok ? base : `${base} (Próximamente)`;
    });
  }

  // Restaurar selección guardada (pero validada)
  let saved = (__savedMachineBySubcat[value] || "").trim();
  if(saved && !allowed.includes(saved)) saved = "";
  if(sel) sel.value = saved;

  // Label + hidden
  const lab = document.getElementById("appMachineLabel");
  const inMid = document.getElementById("machineIdInput");
  if(inMid) inMid.value = saved;
  if(lab) lab.textContent = saved ? `Máquina ${saved}` : "—";

  gateUploadBlock();
}

function onAppMachineChange(selectEl){
  let m = String(selectEl?.value || "").trim();

  const sub = (__currentSubcat || (document.getElementById("subcatInput")?.value || "")).trim().toLowerCase();
  const allowed = APP_ENABLED[sub] || [];

  // ✅ Bloquear lo que no esté habilitado
  if(m && !allowed.includes(m)){
    const allowTxt = allowed.length ? allowed.map(x=>`Máquina ${x}`).join(" y ") : "—";
    alert(`Por ahora solo está habilitado: ${sub === "union" ? "Unión" : "Aplicación"} - ${allowTxt}.`);
    m = "";
    if(selectEl) selectEl.value = "";
  }

  // Guardar por subcat
  if(sub){
    __savedMachineBySubcat[sub] = m;
  }

  // Label + hidden
  const lab = document.getElementById("appMachineLabel");
  if(lab) lab.textContent = m ? `Máquina ${m}` : "—";

  const inMid = document.getElementById("machineIdInput");
  if(inMid) inMid.value = m;

  gateUploadBlock();

  // ✅ Auto-entrar al análisis cuando ya hay subcat + machine_id válidos
  if(__IS_APP_LINE && __IS_UPLOAD_STAGE){
    const mid = (document.getElementById("machineIdInput")?.value || "").trim();
    const ok = (!!sub && !!mid && (APP_ENABLED[sub] || []).includes(mid));
    if(ok){
      const f = document.getElementById("appAutoContinueForm");
      if(f) f.submit();
    }
  }
}

function gateUploadBlock(){
  const uploadBlock = document.getElementById("uploadBlock");
  const hintBlock = document.getElementById("hintBlock");
  if(!uploadBlock || !hintBlock) return;

  const machine = document.getElementById("machineInput")?.value || "";
  const isAplicacion = (__IS_APP_LINE || _cfgStr(CFG.line_key) === "aplicacion");
  const subcat = document.getElementById("subcatInput")?.value || "";

  const mid = document.getElementById("machineIdInput")?.value || "";
  const ok = isAplicacion ? (!!subcat && !!mid) : (!!machine);

  hintBlock.style.display = ok ? "none" : "block";
  uploadBlock.style.display = ok ? "block" : "none";
}

// =========================
// MACHINE (solo Corte)
// =========================
function selectMachine(name, el){
  document.querySelectorAll(".machine-card").forEach(c => c.classList.remove("selected"));
  el.classList.add("selected");

  const label = document.getElementById("machineLabel");
  const input = document.getElementById("machineInput");

  // Corte (HP/THB) - Google Sheets
  const inputGs = document.getElementById("machineInputGs");
  const gsheetWrap = document.getElementById("gsheetHpWrap");

  // ✅ Continuidad (BUSES/MOTOS) - Google Sheets
  const inputGsCont = document.getElementById("machineInputGsCont");
  const gsheetContWrap = document.getElementById("gsheetContWrap");

  if(label) label.textContent = name;
  if(input) input.value = name;

  // =========================
  // ✅ CORTE: Google Sheets (solo HP/THB y solo si estás en corte)
  // =========================
  if(inputGs) inputGs.value = name;

  if(gsheetWrap){
    const isCorte = (window.LINE_START_CFG?.line_key === "corte");
    const m = String(name).toUpperCase();
    gsheetWrap.style.display = (isCorte && (m === "HP" || m === "THB")) ? "block" : "none";
  }

  // =========================
  // ✅ CONTINUIDAD: Google Sheets (BUSES/MOTOS)
  // =========================
  if(inputGsCont) inputGsCont.value = name;

  if(gsheetContWrap){
    const isCont = (window.LINE_START_CFG?.line_key === "continuidad");
    gsheetContWrap.style.display = isCont ? "block" : "none";
  }

  gateUploadBlock();
}



// =========================
// PERIOD
// =========================
function selectPeriod(value, el){
  document.querySelectorAll(".period-card").forEach(c => c.classList.remove("selected"));
  el.classList.add("selected");

  const periodInput = document.getElementById("periodInput");
  if(periodInput) periodInput.value = value;

  const lbl = document.getElementById("periodLabel");
  if(lbl){
    if(value === "day") lbl.textContent = "Días disponibles";
    if(value === "week") lbl.textContent = "Semanas disponibles";
    if(value === "month") lbl.textContent = "Meses disponibles";
  }

  const sel = document.getElementById("periodValueSelect");
  if(sel) sel.value = "";

  hideThbFilters();
  hideInlineResults();
  reloadOptions();
}

function hideThbFilters(){
  const box = document.getElementById("thbFilters");
  const btn = document.getElementById("btnAnalyze");
  if(box) box.style.display = "none";
  if(btn) btn.disabled = true;
}

function hideInlineResults(){
  const box = document.getElementById("inlineResults");
  if(box) box.style.display = "none";
  destroyPareto();
  destroyTerminalUsage();
  hideProdHour();

  const hpLine = document.getElementById("hpTimesLine");
  if(hpLine){
    hpLine.style.display = "none";
    hpLine.textContent = "";
  }
}

// =========================
// HP: Línea resumen de tiempos
// =========================
function renderHpTimesLine(_data){
  const el = document.getElementById("hpTimesLine");
  if(!el) return;
  el.textContent = "";
  el.style.display = "none";
}

async function reloadOptions(){
  const period = document.getElementById("periodInput")?.value || "day";
  const sel = document.getElementById("periodValueSelect");
  if(!sel) return;

  sel.innerHTML = `<option value="">Cargando...</option>`;

  const filename = _cfgStr(CFG.filename);

  // ✅ usar machine real (hidden) y NO CFG.machine
  const machine = getCurrentMachine();

  function fmtDateSlash(label){
    const s = String(label ?? "").trim();
    if(/^\d{4}-\d{2}-\d{2}$/.test(s)) return s.replaceAll("-", "/");
    return s;
  }

  // ✅ machine_id (para Aplicación/Unión)
  const mu = getCurrentMachineU();
  let mid = "";
  if(mu === "APLICACION" || mu === "UNION" || __IS_APP_LINE){
    mid = (
      document.getElementById("machineIdInput2")?.value ||
      document.getElementById("machineIdInput")?.value ||
      _cfgStr(CFG.machine_id) ||
      ""
    ).trim();
  }

  try{
    if(!URLS.period_options){
      sel.innerHTML = `<option value="">Error: URL period_options no configurada</option>`;
      return;
    }

    const url =
      `${URLS.period_options}?filename=${encodeURIComponent(filename)}` +
      `&analysis_key=${encodeURIComponent(_cfgStr(CFG.line_key))}` +   // ✅ NUEVO
      `&machine=${encodeURIComponent(machine)}` +
      `&period=${encodeURIComponent(period)}` +
      (mid ? `&machine_id=${encodeURIComponent(mid)}` : ``) +
      `&_=${Date.now()}`;

    const r = await fetch(url, { cache: "no-store" });
    const data = await r.json();

    const opts = data.options || [];
    if(opts.length === 0){
      sel.innerHTML = `<option value="">No hay registros para este periodo</option>`;
      return;
    }

    sel.innerHTML =
      `<option value="">Selecciona...</option>` +
      opts.map(o => {
        const v = String(o.value ?? "");
        const lab = fmtDateSlash(o.label ?? o.value ?? "");
        return `<option value="${escHtml(v)}">${escHtml(lab)}</option>`;
      }).join("");

  }catch(e){
    sel.innerHTML = `<option value="">Error cargando opciones</option>`;
  }
}

// ✅ Auto-run (debounce)
let _autoRunTimer = null;
function scheduleAutoRun(){
  const form = document.getElementById("runForm");
  if(!form) return;
  if(!periodValueOk()) return;

  if(_autoRunTimer) clearTimeout(_autoRunTimer);
  _autoRunTimer = setTimeout(()=>runInlineAnalysis(form, { scroll:false, preserveScroll:true }), 250);
}

function setTimeApply(flag){
  const el = document.getElementById("timeApply");
  if(el) el.value = flag ? "1" : "0";
}

function setTimeInputsVisible(show){
  const t0 = document.getElementById("timeStart");
  const t1 = document.getElementById("timeEnd");
  if(!t0 || !t1) return;

  const b0 = t0.closest(".select-box");
  const b1 = t1.closest(".select-box");

  if(b0) b0.style.display = show ? "block" : "none";
  if(b1) b1.style.display = show ? "block" : "none";

  t0.required = !!show;
  t1.required = !!show;

  if(!show){
    t0.value = "";
    t1.value = "";
  }
}

async function onPeriodValueChange(){
  hideInlineResults();
  setTimeApply(false);

  // ✅ usar machine real (hidden)
  const machine = getCurrentMachineU();

  const period = document.getElementById("periodInput")?.value || "day";
  const isDay = (period === "day");
  const periodValue = document.getElementById("periodValueSelect")?.value || "";

  if(!periodValue || periodValue === "No hay registros para este periodo"){
    hideThbFilters();
    return;
  }

  const filename = _cfgStr(CFG.filename);

    // ✅ Ciclo/Aplicadores: no usa subcat/machine_id ni operadores
  if(__IS_CICLO_LINE){
    const opSel = document.getElementById("operatorSelect");
    const box = document.getElementById("thbFilters");
    const btn = document.getElementById("btnAnalyze");

    if(opSel){
      opSel.innerHTML = `<option value="General">General</option>`;
      opSel.value = "General";
    }

    setTimeInputsVisible(false);   // no horas
    if(box) box.style.display = "block";
    if(btn) btn.disabled = false;

    // opcional: auto-run al elegir el periodo
    scheduleAutoRun();
    return;
  }




  // =========================
  // APLICACION / UNION
  // =========================
  if(machine === "APLICACION" || machine === "UNION"){
    try{
      const subcat = document.getElementById("subcatInput2")?.value || _cfgStr(CFG.subcat) || "";
      const mid = (document.getElementById("machineIdInput2")?.value || _cfgStr(CFG.machine_id) || "").trim();

      if(!URLS.app_filter_options){
        hideThbFilters();
        return;
      }

      const url =
        `${URLS.app_filter_options}?filename=${encodeURIComponent(filename)}` +
        `&machine=${encodeURIComponent(machine)}` +
        `&subcat=${encodeURIComponent(subcat)}` +
        `&machine_id=${encodeURIComponent(mid)}` +
        `&period=${encodeURIComponent(period)}` +
        `&period_value=${encodeURIComponent(periodValue)}` +
        `&_=${Date.now()}`;

      const r = await fetch(url, { cache: "no-store" });
      const data = await r.json();

      const ops = data.operators || [];
      const minT = (Object.prototype.hasOwnProperty.call(data, "min_time")) ? (data.min_time ?? "") : "00:00";
      const maxT = (Object.prototype.hasOwnProperty.call(data, "max_time")) ? (data.max_time ?? "") : "23:59";

      const opSel = document.getElementById("operatorSelect");
      const t0 = document.getElementById("timeStart");
      const t1 = document.getElementById("timeEnd");
      const box = document.getElementById("thbFilters");
      const btn = document.getElementById("btnAnalyze");

      if(!opSel || !t0 || !t1 || !box || !btn) return;

      const opsClean = ops
        .map(o => String(o ?? "").trim())
        .filter(o => o && o.toLowerCase() !== "general");

      opSel.innerHTML =
        `<option value="General">General</option>` +
        opsClean.map(o => `<option value="${o}">${o}</option>`).join("");

      opSel.value = "General";

      setTimeInputsVisible(isDay);

      if(isDay){
        t0.value = (minT || "00:00");
        t1.value = (maxT || "23:59");
      }

      box.style.display = "block";
      btn.disabled = true;

      function validate(){
        const ok = periodValueOk() && opSel.value && (!isDay || (t0.value && t1.value));
        btn.disabled = !ok;
        if(ok) scheduleAutoRun();
      }

      opSel.onchange = () => { setTimeApply(false); validate(); };

      const onTimeChange = () => { setTimeApply(true); validate(); };
      t0.onchange = onTimeChange; t0.oninput = onTimeChange;
      t1.onchange = onTimeChange; t1.oninput = onTimeChange;

      validate();

    }catch(e){
      hideThbFilters();
    }
    return;
  }

  // =========================
  // THB / HP
  // =========================
  if(machine !== "THB" && machine !== "HP"){
    hideThbFilters();
    return;
  }

  try{
    if(!URLS.thb_filter_options){
      hideThbFilters();
      return;
    }

    const url =
      `${URLS.thb_filter_options}?filename=${encodeURIComponent(filename)}` +
      `&machine=${encodeURIComponent(machine)}` +
      `&period=${encodeURIComponent(period)}` +
      `&period_value=${encodeURIComponent(periodValue)}` +
      `&_=${Date.now()}`;

    const r = await fetch(url, { cache: "no-store" });
    const data = await r.json();

    const ops = data.operators || [];
    const minT = (Object.prototype.hasOwnProperty.call(data, "min_time")) ? (data.min_time ?? "") : "00:00";
    const maxT = (Object.prototype.hasOwnProperty.call(data, "max_time")) ? (data.max_time ?? "") : "23:59";

    const opSel = document.getElementById("operatorSelect");
    const t0 = document.getElementById("timeStart");
    const t1 = document.getElementById("timeEnd");
    const box = document.getElementById("thbFilters");
    const btn = document.getElementById("btnAnalyze");

    if(!opSel || !t0 || !t1 || !box || !btn) return;

    const opsClean = ops
      .map(o => String(o ?? "").trim())
      .filter(o => o && o.toLowerCase() !== "general");

    opSel.innerHTML =
      `<option value="General">General</option>` +
      opsClean.map(o => `<option value="${o}">${o}</option>`).join("");

    opSel.value = "General";

    setTimeInputsVisible(isDay);

    if(isDay){
      t0.value = (minT || "00:00");
      t1.value = (maxT || "23:59");
    }

    box.style.display = "block";
    btn.disabled = true;

    function validate(){
      const ok = periodValueOk() && opSel.value && (!isDay || (t0.value && t1.value));
      btn.disabled = !ok;
      if(ok) scheduleAutoRun();
    }

    opSel.onchange = () => { setTimeApply(false); validate(); };

    const onTimeChange = () => { setTimeApply(true); validate(); };
    t0.onchange = onTimeChange; t0.oninput = onTimeChange;
    t1.onchange = onTimeChange; t1.oninput = onTimeChange;

    validate();

  }catch(e){
    hideThbFilters();
  }
}

// =========================
// Helpers UI
// =========================
function escHtml(s){
  return String(s ?? "")
    .replaceAll("&","&amp;").replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function getPeriodValue(){
  return String(document.getElementById("periodValueSelect")?.value || "").trim();
}
function periodValueOk(){
  if(__IS_CICLO_LINE) return true;
  const pv = getPeriodValue();
  return !!pv && pv !== "No hay registros para este periodo";
}

function _parseFirstNumFromEl(el){
  const t = el ? (el.textContent || "") : "";

  // 1) Primero intenta HH:MM:SS o HH:MM
  const mt = String(t).match(/\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b/);
  if(mt){
    const hh = Number(mt[1]) || 0;
    const mm = Number(mt[2]) || 0;
    const ss = Number(mt[3]) || 0;
    return hh + (mm/60) + (ss/3600); // horas decimales (para cálculo)
  }

  // 2) Si no hay horas, toma el primer número (con coma o punto)
  const m = String(t || "").replace(",", ".").match(/-?\d+(?:\.\d+)?/);
  return m ? Number(m[0]) : NaN;
}


function _stripPctText(s){
  return String(s || "").replace(/\(\s*\d+(?:\.\d+)?\s*%\s*\)/g, "").trim();
}

function _fmtHHMMFromHoursDec(h){
  // truncado a minuto (sin redondeo)
  const mins = Math.max(0, Math.trunc((Number(h)||0) * 60));
  const hh = Math.floor(mins / 60);
  const mm = mins % 60;
  return String(hh).padStart(2,"0") + ":" + String(mm).padStart(2,"0");
}


function _fmtPct1(x){
  const r = Math.round((Number(x)||0) * 10) / 10;
  if(!Number.isFinite(r)) return "0%";
  if(Math.abs(r - Math.round(r)) < 1e-9) return `${Math.round(r)}%`;
  return `${r.toFixed(1)}%`;
}

function _fmtPctPartsTo100(prodSec, otherSec, mealSec){
  const p = Math.max(0, Number(prodSec)  || 0);
  const o = Math.max(0, Number(otherSec) || 0);
  const m = Math.max(0, Number(mealSec)  || 0);
  const base = p + o + m;

  if(base <= 0){
    return { prodPct: "0%", otherPct: "0%", mealPct: "0%" };
  }

  let pp = Math.round(100 * p / base);
  let op = Math.round(100 * o / base);
  let mp = Math.round(100 * m / base);

  // Ajusta el último “existente” para cerrar 100
  let key = (m > 0) ? "meal" : (o > 0) ? "other" : "prod";
  const sum = pp + op + mp;
  const d = 100 - sum;

  if(key === "meal") mp += d;
  else if(key === "other") op += d;
  else pp += d;

  // clamps
  if(pp < 0) pp = 0;
  if(op < 0) op = 0;
  if(mp < 0) mp = 0;

  return { prodPct: `${pp}%`, otherPct: `${op}%`, mealPct: `${mp}%` };
}



// ✅ Ajusta No registrado (y %) en la card de HP para que cierre con Horas Disponibles
function adjustHpNoRegistradoUI(){
  const cards = Array.from(document.querySelectorAll("#inlineKpis .kpi-card"));
  const getCard = (title)=> cards.find(c =>
    (c.querySelector(".kpi-title")?.textContent || "").trim() === title
  );

  const hdCard   = getCard("Horas Disponibles");
  const effCard  = getCard("Tiempo Trabajado");
  const otroCard = getCard("Tiempos Perdidos");
  if(!hdCard || !effCard || !otroCard) return;

  const hdEl   = hdCard.querySelector(".kpi-value");
  const effEl  = effCard.querySelector(".kpi-value");
  const otroEl = otroCard.querySelector(".kpi-split-left .kpi-value") || otroCard.querySelector(".kpi-value");
  if(!hdEl || !effEl || !otroEl) return;

  const hdVal   = _parseFirstNumFromEl(hdEl);
  const effVal  = _parseFirstNumFromEl(effEl);
  const otroVal = _parseFirstNumFromEl(otroEl);
  if(!Number.isFinite(hdVal) || !Number.isFinite(effVal) || !Number.isFinite(otroVal)) return;

  // =========================
  // 1) No registrado base (lo que falta para llegar a Horas Disponibles)
  // =========================
  let nr = hdVal - (effVal + otroVal);
  if(!Number.isFinite(nr) || nr < 0) nr = 0;
  nr = _trunc2(nr);

  // =========================
  // 2) ✅ MOVER EXCEDENTE de Parada 105 → No registrado (SOLO VISUAL)
  //    Si (eff + otro) > hd, el exceso lo sacamos de Parada 105 (hasta donde alcance)
  // =========================
  const overflow = Math.max(0, _trunc2((effVal + otroVal) - hdVal)); // horas decimales
  let moved = 0;

  // localizar bloques
  const blocks = Array.from(otroCard.querySelectorAll(".kpi-sub-block"));
  const findBlock = (labelLower)=> blocks.find(b =>
    (b.querySelector(".kpi-sub-label")?.textContent || "").trim().toLowerCase() === labelLower
  );

  const b105 = findBlock("parada 105");
  let bNR  = findBlock("no registrado");

  // asegurar bloque NR
  if(!bNR){
    const row = otroCard.querySelector(".kpi-sub-row");
    if(row){
      const div = document.createElement("div");
      div.className = "kpi-sub-divider";
      row.appendChild(div);

      bNR = document.createElement("div");
      bNR.className = "kpi-sub-block";
      bNR.innerHTML = `
        <div class="kpi-sub-label">No registrado</div>
        <div class="kpi-sub-value-big">0.00</div>
      `;
      row.appendChild(bNR);
    }
  }

  // si existe parada 105 y hay overflow, lo movemos
  if(b105 && overflow > 0){
    const v105El = b105.querySelector(".kpi-sub-value-big");
    const v105 = _parseFirstNumFromEl(v105El);

    if(Number.isFinite(v105) && v105 > 0){
      moved = Math.min(v105, overflow);
      const v105New = _trunc2(Math.max(0, v105 - moved));

      if(v105El) v105El.textContent = _fmt2(v105New);
      nr = _trunc2(nr + moved);
    }
  }

  // pintar NR final
  const nrTxt = _fmt2(nr);
  const nrValEl = bNR?.querySelector(".kpi-sub-value-big");
  if(nrValEl) nrValEl.textContent = nrTxt;

  // =========================
  // 3) % (base = Horas Disponibles)
  // =========================
  const base = hdVal;
  if(base > 0){
    const pctEff  = _fmtPct1(100 * effVal  / base);
    const pctOtro = _fmtPct1(100 * otroVal / base);

    const effTimeTxt = _stripPctText(effEl.textContent);
    effEl.innerHTML = `${escHtml(effTimeTxt)} <span class="kpi-pct">(${escHtml(pctEff)})</span>`;

    const otroTimeTxt = _stripPctText(otroEl.textContent);
    otroEl.innerHTML = `${escHtml(otroTimeTxt)} <span class="kpi-pct">(${escHtml(pctOtro)})</span>`;
  }
}







function makeCard(title, value, cls, spanClass){
  const card = document.createElement("div");
  card.className = `kpi-card ${cls || ""} ${spanClass || ""}`;

  const raw = String(value ?? "");
  const parts = raw.split("||");

  const mainRaw = (parts[0] ?? "").trim();       // ✅ ahora sí existe
  const mainDec = _replaceTimesToHourDec(mainRaw);
  const sub  = parts.slice(1).join("||").trim();

  if(sub){
    const pipeParts = sub
      .split("|")
      .map(x => String(x || "").trim())
      .filter(Boolean);

    let v105 = "";
    let v104 = "";
    let nrVal = "";
    let pctVal = "";

    const pctRe = /^\(?\s*\d+(\.\d+)?\s*%\s*\)?$/i;

    for(const seg of pipeParts){
      const s = seg.trim();

      // ✅ excluir 000 / Desconocido SOLO del desglose
      if(/^(\(?\s*)?(?:parada\s*)?000\s*:/i.test(s) || /desconocido/i.test(s)){
        continue;
      }

      // % (14.2%)
      if(pctRe.test(s)){
        pctVal = s.replace(/[()]/g, "").trim();
        continue;
      }

      // Parada 105 / 105
      let m = s.match(/^(?:parada\s*)?105\s*:\s*(.+)$/i);
      if(m){
        v105 = (m[1] || "").trim();
        continue;
      }

      // Parada 104 / 104
      m = s.match(/^(?:parada\s*)?104\s*:\s*(.+)$/i);
      if(m){
        v104 = (m[1] || "").trim();
        continue;
      }

      // No registrado
      m = s.match(/^no\s*registrado\s*:\s*(.+)$/i);
      if(m){
        nrVal = (m[1] || "").trim();
        continue;
      }

      // Back-compat
      if(!v105) v105 = s;
    }

    // defaults
    if(!v105) v105 = "0.00";
    if(!v104) v104 = "0.00";
    if(!nrVal) nrVal = "0.00";

    // ✅ convertir HH:MM -> horas decimales (si venían así)
    v105  = _replaceTimesToHourDec(v105);
    v104  = _replaceTimesToHourDec(v104);
    nrVal = _replaceTimesToHourDec(nrVal);

    const mainHtml = pctVal
      ? `${escHtml(mainDec)} <span class="kpi-pct">(${escHtml(pctVal)})</span>`
      : `${escHtml(mainDec)}`;

    card.innerHTML = `
      <div class="kpi-title">${escHtml(title)}</div>

      <div class="kpi-split">
        <div class="kpi-split-left">
          <div class="kpi-value">${mainHtml}</div>
        </div>

        <div class="kpi-split-right">
          <div class="kpi-sub-row">
            <div class="kpi-sub-block">
              <div class="kpi-sub-label">Parada 105</div>
              <div class="kpi-sub-value-big">${escHtml(v105)}</div>
            </div>

            <div class="kpi-sub-divider"></div>

            <div class="kpi-sub-block">
              <div class="kpi-sub-label">No registrado</div>
              <div class="kpi-sub-value-big">${escHtml(nrVal)}</div>
            </div>

            <div class="kpi-sub-divider"></div>

            <div class="kpi-sub-block">
              <div class="kpi-sub-label">Parada 104</div>
              <div class="kpi-sub-value-big">${escHtml(v104)}</div>
            </div>
          </div>
        </div>
      </div>
    `;
    return card;
  }

  // Simple: puede venir "3.30 (50.8%)"
  let mainHtmlSimple = escHtml(_replaceTimesToHourDec(mainRaw));

  const mPct = String(mainRaw || "").match(/^(.*?)(\s*\(\s*\d+(?:\.\d+)?\s*%\s*\)\s*)$/);
  if(mPct){
    const left = (mPct[1] || "").trim();
    const pct  = (mPct[2] || "").trim();
    mainHtmlSimple = `${escHtml(_replaceTimesToHourDec(left))} <span class="kpi-pct">${escHtml(pct)}</span>`;
  }

  card.innerHTML = `
    <div class="kpi-title">${escHtml(title)}</div>
    <div class="kpi-value">${mainHtmlSimple}</div>
  `;
  return card;
}












function periodTitle(period){
  if(period === "day") return "Indicadores diarios";
  if(period === "week") return "Indicadores semanales";
  if(period === "month") return "Indicadores mensuales";
  return "Indicadores";
}

// =========================
// ✅ FIX SCROLL: Capturar/Restaurar ancla visible (tarjetas hora/hora)
// =========================
function _captureHourAnchor(){
  const cards = Array.from(document.querySelectorAll("#prodHourGrid .ph-card"));
  for(const c of cards){
    const r = c.getBoundingClientRect();
    if(r.bottom > 0 && r.top < window.innerHeight){
      const key = (c.querySelector(".ph-hour")?.textContent || "").trim();
      return { type:"hour", key, top:r.top, y:window.scrollY };
    }
  }
  return { type:"y", y:window.scrollY };
}

function _restoreHourAnchor(a){
  if(!a) return;

  if(a.type === "hour" && a.key){
    const cards = Array.from(document.querySelectorAll("#prodHourGrid .ph-card"));
    const target = cards.find(c => ((c.querySelector(".ph-hour")?.textContent || "").trim() === a.key));
    if(target){
      const r = target.getBoundingClientRect();
      const delta = r.top - (a.top || 0);
      if(delta) window.scrollBy(0, delta);
      return;
    }
  }
  if(typeof a.y === "number") window.scrollTo(0, a.y);
}

// =========================
// DICCIONARIO CÓDIGOS -> DESCRIPCIÓN
// =========================
const STOP_CODE_DESC = {
  "101":"Reuniones/Capacitación",
  "104":"Pausas activas",
  "105":"Almuerzo/Desayuno/Refrigerio horas extras",
  "106":"Baños/Pausa para tomar Agua",
  "108":"Mantenimiento Preventivo",
  "110":"Orden y Aseo General de Puesto",
  "111":"Puestas a punto/Pruebas de calidad",
  "201":"Falta de energía/Aire comprimido",
  "203":"Cambio de chipas o carretes",
  "204":"Parada por daño en el aplicador",
  "205":"Parada por daño máquina de aplicación/uniones/manguera/corte/video jet",
  "206":"Cables enredados/Empalmar cables",
  "219":"Parada por daño en software o hardware/ PLC",
  "221":"Parada por falta de insumos o materiales",
  "222":"Problema/Falta de Información (Planillas, Desarrollos o Planos/tarjetas, tableros)",
  "224":"Parada por materiales defectuosos o fuera de especificación",
  "226":"Faltante de Circuito/ Error en el Circuito",
  "235":"Accidente/incidentes",
  "237":"Tiempo espera mantenimiento",
  "238":"Ajuste de aplicador-Adaptaciones",
  "240":"corte de terminal manual",
  "241":"Cambio de aplicador",
  "242": 'Accesorios/Fusileras/Separacion de RACK',
  "243": 'Cambio de maquina',
  "000":"Desconocido"

};
// =========================
// Intervalos especiales (sin código numérico) — normalizados
// =========================
const OTHER_INTERVAL_DESC = {
  "finalizar→inicio": "Finalizar → Inicio",
  "reprocesoinicio→reprocesofin": "Reproceso (inicio → fin)",
  "pruebatensioninicio→pruebatensionfin": "Prueba de tensión (inicio → fin)",
};



// =========================
// PARETO (PARADAS) ✅ THB y HP
// =========================
let _paretoChart = null;

function paretoTitle(period){
  if(period === "day") return "Análisis de Pareto de Causas de Parada por Día";
  if(period === "week") return "Análisis de Pareto de Causas de Parada por Semana";
  if(period === "month") return "Análisis de Pareto de Causas de Parada por Mes";
  return "Análisis de Pareto de Causas de Parada";
}

function secToHHMM(sec){
  // ✅ ahora devuelve horas decimales (2 decimales, truncado)
  return _fmtHourDecFromSec(sec);
}

function secToHHMMSS(sec){
  // ✅ ahora también devuelve horas decimales
  return _fmtHourDecFromSec(sec);
}


function _trunc2(x){
  const n = Number(x);
  if(!Number.isFinite(n)) return 0;
  return Math.trunc(n * 100) / 100; // ✅ sin redondeo
}

function _fmtHourDecPartsToOneHour(prodSec, otherSec, mealSec){
  // Trabaja en HORAS (no strings) y fuerza cierre visual a 1.00
  const p = Math.max(0, Number(prodSec) || 0) / 3600;
  const o = Math.max(0, Number(otherSec) || 0) / 3600;
  const m = Math.max(0, Number(mealSec) || 0) / 3600;

  // Redondea dos y el tercero lo ajusta
  let p2 = _round2(p);
  let o2 = _round2(o);

  // Remanente para cerrar 1.00
  let m2 = _round2(1 - (p2 + o2));

  // clamps por si por redondeo queda -0.01 o 1.01
  if(m2 < 0){
    // quítaselo a "Otro" primero (o a Efectivo si prefieres)
    o2 = _round2(o2 + m2);
    m2 = 0;
  }
  if(o2 < 0){ o2 = 0; }

  // segunda pasada por si quedó 1.01 etc
  const sum = _round2(p2 + o2 + m2);
  if(sum !== 1){
    // ajusta el último (comida) para cerrar exacto
    m2 = _round2(m2 + (1 - sum));
    if(m2 < 0) m2 = 0;
  }

  return {
    prodStr: _fmt2(p2),
    otherStr: _fmt2(o2),
    mealStr: _fmt2(m2),
  };
}

function _fmtHourDecPartsToTotal(prodSec, otherSec, mealSec){
  const p = Math.max(0, Number(prodSec)  || 0) / 3600;
  const o = Math.max(0, Number(otherSec) || 0) / 3600;
  const m = Math.max(0, Number(mealSec)  || 0) / 3600;

  // total real y total que vas a mostrar
  const totalRaw = p + o + m;
  const total2 = _round2(totalRaw);

  // redondea 2 y ajusta el último para que SUMA == total2
  let p2 = _round2(p);
  let o2 = _round2(o);
  let m2 = _round2(m);

  // elige cuál ajustar: prioridad al último que exista (>0), si no, comida, si no, otro, si no, efectivo
  let adjustKey = "meal";
  if(m2 > 0 || m > 0) adjustKey = "meal";
  else if(o2 > 0 || o > 0) adjustKey = "other";
  else adjustKey = "prod";

  const sum2 = _round2(p2 + o2 + m2);
  const delta = _round2(total2 - sum2);

  if(adjustKey === "meal") m2 = _round2(m2 + delta);
  if(adjustKey === "other") o2 = _round2(o2 + delta);
  if(adjustKey === "prod")  p2 = _round2(p2 + delta);

  // clamps por si queda -0.01 por redondeos extremos
  if(p2 < 0) p2 = 0;
  if(o2 < 0) o2 = 0;
  if(m2 < 0) m2 = 0;

  // segunda pasada por seguridad
  const sum3 = _round2(p2 + o2 + m2);
  if(sum3 !== total2){
    // fuerza el ajuste al mismo key
    const d2 = _round2(total2 - sum3);
    if(adjustKey === "meal") m2 = _round2(m2 + d2);
    if(adjustKey === "other") o2 = _round2(o2 + d2);
    if(adjustKey === "prod")  p2 = _round2(p2 + d2);
  }

  return {
    totalStr: _fmt2(total2),
    prodStr:  _fmt2(p2),
    otherStr: _fmt2(o2),
    mealStr:  _fmt2(m2),
    total2
  };
}


function _round2(x){
  const n = Number(x);
  if(!Number.isFinite(n)) return 0;
  return Math.round(n * 100) / 100;
}
function _fmt2(x){
  return _round2(x).toFixed(2);
}


function _fmtHourDecFromSec(sec){
  const s = Math.max(0, Number(sec) || 0);
  const h = s / 3600;
  return _trunc2(h).toFixed(2); // "9.13"
}
function _fmtHourDecPartsRaw(prodSec, otherSec, mealSec){
  return {
    prodStr:  _fmtHourDecFromSec(prodSec),
    otherStr: _fmtHourDecFromSec(otherSec),
    mealStr:  _fmtHourDecFromSec(mealSec),
  };
}


function _fmtHourDecFromHHMM(str){
  const t = String(str || "").trim();
  // acepta "H:MM", "HH:MM", "HH:MM:SS" (si viene)
  const m = t.match(/^(\d{1,2})\s*:\s*(\d{1,2})(?:\s*:\s*(\d{1,2}))?$/);
  if(!m) return null;
  const hh = Number(m[1]) || 0;
  const mm = Number(m[2]) || 0;
  const ss = Number(m[3]) || 0;
  const dec = hh + (mm/60) + (ss/3600);
  return _trunc2(dec).toFixed(2);
}



// ✅ convierte cualquier texto que contenga HH:MM o HH:MM:SS a horas decimales
function _replaceTimesToHourDec(text){
  const s = String(text ?? "");

  // HH:MM:SS primero
  let out = s.replace(/\b(\d{1,2}):(\d{2}):(\d{2})\b/g, (_,h,m,ss)=>{
    const dec = (Number(h)||0) + (Number(m)||0)/60 + (Number(ss)||0)/3600;
    return _trunc2(dec).toFixed(2);
  });

  // luego HH:MM (que no venga seguido de :SS)
  out = out.replace(/\b(\d{1,2}):(\d{2})\b(?!:\d{2})/g, (_,h,m)=>{
    const dec = (Number(h)||0) + (Number(m)||0)/60;
    return _trunc2(dec).toFixed(2);
  });

  return out;
}


function _destroyParetoChartOnly(){
  if(_paretoChart){
    try{ _paretoChart.destroy(); }catch(e){}
    _paretoChart = null;
  }
}

function destroyPareto(){
  _destroyParetoChartOnly();

  const wrap = document.getElementById("paretoChartWrap");
  if(wrap) wrap.style.display = "none";

  const lw = document.getElementById("paretoLegendWrap");
  const lb = document.getElementById("paretoLegendBody");
  if(lb) lb.innerHTML = "";
  if(lw) lw.style.display = "none";

  const titleEl = document.getElementById("inlineParetoTitle");
  if(titleEl) titleEl.style.display = "none";
}

// ✅ ARREGLADO: ahora llena también % Acum. (4ta columna)
function renderParetoLegend(labels, durSec, cumPct){
  const wrap = document.getElementById("paretoLegendWrap");
  const body = document.getElementById("paretoLegendBody");
  if(!wrap || !body) return;

  body.innerHTML = "";

  if(!labels || !labels.length){
    wrap.style.display = "none";
    return;
  }

  for(let i=0; i<labels.length; i++){
    const code = String(labels[i] ?? "").trim();
    const desc = STOP_CODE_DESC[code] || code || "—";
    const dur  = secToHHMM(durSec?.[i]);
    const cp   = Number(cumPct?.[i] ?? 0) || 0;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${escHtml(code)}</b></td>
      <td>${escHtml(desc)}</td>
      <td style="text-align:right;"><b>${escHtml(dur)}</b></td>
      <td style="text-align:right;"><b>${escHtml(cp)}</b>%</td>
    `;
    body.appendChild(tr);
  }

  wrap.style.display = "block";
}

function renderParetoParadas(result){
  try{
    const pareto = result?.pareto || null;

    const titleEl = document.getElementById("inlineParetoTitle");
    const wrap = document.getElementById("paretoChartWrap");
    const canvas = document.getElementById("paretoChart");

    if(!titleEl || !wrap || !canvas) return;

    let labels = pareto?.labels || [];
    let durSec = pareto?.dur_sec || [];
    let cumPct = pareto?.cum_pct || [];

    // ✅ FILTRAR: excluir "000" / "Desconocido" SOLO en Pareto
    const filt = [];
    for(let i=0; i<labels.length; i++){
      const code = String(labels[i] ?? "").trim();
      const sec  = Number(durSec[i] ?? 0) || 0;
      if(sec <= 0) continue;

      // excluir 000 y/o texto "desconocido"
      const desc = (STOP_CODE_DESC[code] || code || "").toString().toLowerCase().trim();
      if(code === "000" || desc === "desconocido") continue;

      // ✅ excluir 105 del pareto (HP y THB)
      if(code === "105") continue;

      filt.push({ code, sec });
    }

    if(filt.length === 0){
      destroyPareto();
      return;
    }

    // reconstruir arrays ya filtrados
    labels = filt.map(x => x.code);
    durSec = filt.map(x => x.sec);

    // ✅ recalcular cumPct (ya NO usamos el del backend porque cambió el total)
    const totalSec = durSec.reduce((a,b)=>a+(Number(b)||0),0);
    let run = 0;
    cumPct = durSec.map(s=>{
      run += (Number(s)||0);
      return totalSec > 0 ? Math.round((run/totalSec)*100) : 0;
    });


    if(!Array.isArray(labels) || !labels.length || !Array.isArray(durSec) || !durSec.length){
      destroyPareto();
      return;
    }

    titleEl.textContent = paretoTitle(String(result?.period || ""));
    titleEl.style.display = "block";

    // normalizar acumulado: si viene 0..1 => 0..100
    const maxCum = Math.max(...cumPct.map(x => Number(x)||0), 0);
    if(maxCum <= 1.01) cumPct = cumPct.map(x => Math.round((Number(x)||0) * 100));
    else cumPct = cumPct.map(x => Math.round(Number(x)||0));

    renderParetoLegend(labels, durSec, cumPct);

    if(typeof Chart === "undefined"){
      wrap.style.display = "none";
      return;
    }

    wrap.style.display = "block";
    _destroyParetoChartOnly();

    const durMin = durSec.map(x => (Number(x) || 0) / 60.0);
    const durSecLocal = durSec.map(x => Number(x) || 0);

    _paretoChart = new Chart(canvas.getContext("2d"), {
      data: {
        labels,
        datasets: [
          { type: "bar",  label: "Duración (min)", data: durMin, yAxisID: "y" },
          { type: "line", label: "Acumulado (%)",  data: cumPct, yAxisID: "y1", tension: 0.25, pointRadius: 2 }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: function(ctx){
                if(ctx.dataset && ctx.dataset.type === "bar"){
                  const s = durSecLocal[ctx.dataIndex] ?? 0;
                  return "Duración: " + secToHHMM(s);
                }
                return "Acumulado: " + (ctx.parsed?.y ?? ctx.raw) + "%";
              }
            }
          }
        },
        scales: {
          y:  { beginAtZero: true, title: { display: true, text: "Duración (min)" } },
          y1: { beginAtZero: true, min: 0, max: 100, position: "right",
                grid: { drawOnChartArea: false },
                title: { display: true, text: "Acumulado (%)" } }
        }
      }
    });

  }catch(e){
    console.error("renderParetoParadas error:", e);
  }
}

// =========================
// TERMINALES (APLICACION/UNION) ✅
// =========================
let _terminalUsageChart = null;

function destroyTerminalUsage(){
  const t = document.getElementById("terminalUsageTitle");
  const w = document.getElementById("terminalUsageWrap");
  const g = document.getElementById("terminalUsageGrid");
  if(g) g.innerHTML = "";
  if(t) t.style.display = "none";
  if(w) w.style.display = "none";
}

function fmtIntEs(n){
  return (Number(n)||0).toLocaleString("es-CO");
}
function fmtMetersEs(m){
  const v = Number(m)||0;
  return v.toLocaleString("es-CO", {minimumFractionDigits:2, maximumFractionDigits:2}) + " m";
}

function renderTerminalUsage(result){
  // ✅ acepta "UNION (4 libros)" / "APLICACION (...)" también
  const mRaw = String(result?.machine || "").toUpperCase().trim();
  const m = mRaw.split("(")[0].trim(); // "UNION (4 libros)" -> "UNION"

  if(m !== "APLICACION" && m !== "UNION"){
    destroyTerminalUsage();
    return;
  }

  const tu = result?.terminal_usage || null;
  const labels = tu?.labels || [];
  const manual = tu?.manual || [];
  const carrete = tu?.carrete || [];
  const otros = tu?.otros || [];
  const total = tu?.total || [];

  const titleEl = document.getElementById("terminalUsageTitle");
  const wrap = document.getElementById("terminalUsageWrap");
  const grid = document.getElementById("terminalUsageGrid");
  if(!titleEl || !wrap || !grid) return;

  if(!Array.isArray(labels) || labels.length === 0){
    destroyTerminalUsage();
    return;
  }

  const sumCarrete = (carrete || []).reduce((a,b)=>a+(Number(b)||0),0);
  const sumOtros   = (otros   || []).reduce((a,b)=>a+(Number(b)||0),0);
  const totalOnly  = (sumCarrete === 0 && sumOtros === 0);

  // ✅ título correcto para tu caso (solo pulsos por terminal)
  titleEl.textContent = totalOnly ? "Crimpados por terminal" : "Terminales más usadas (Manual vs Carrete)";
  titleEl.style.display = "block";
  wrap.style.display = "block";
  grid.innerHTML = "";

  for(let i=0; i<labels.length; i++){
    const term = String(labels[i] ?? "").trim();

    const man  = Number(manual[i] || 0);
    const car  = Number(carrete[i] || 0);
    const oth  = Number(otros[i] || 0);
    const tot  = Number(total[i] || (man+car+oth));

    const pm = tot > 0 ? Math.round((man/tot)*100) : 0;
    const pc = tot > 0 ? Math.round((car/tot)*100) : 0;
    const po = tot > 0 ? Math.max(0, 100 - pm - pc) : 0;

    const card = document.createElement("div");
    card.className = "tu-card";

    const badgesHtml = totalOnly ? "" : `
      <div class="tu-badges">
        <span class="tu-badge">Manual: ${escHtml(fmtIntEs(man))} (${pm}%)</span>
        <span class="tu-badge">Carrete: ${escHtml(fmtIntEs(car))} (${pc}%)</span>
        ${ (sumOtros > 0) ? `<span class="tu-badge">Otros: ${escHtml(fmtIntEs(oth))} (${po}%)</span>` : `` }
      </div>
    `;

    card.innerHTML = `
      <div class="tu-head">
        <div>
          <div class="tu-terminal">${escHtml(term)}</div>
        </div>
        <div>
          <div class="tu-total">${escHtml(fmtIntEs(tot))}</div>
          <div class="tu-sub" style="text-align:right;">Crimpados</div>
        </div>
      </div>
      ${badgesHtml}
    `;

    grid.appendChild(card);
  }
}


// =========================
// ✅ Pareto Aplicación/Unión (usa result.pareto.items)
// =========================
function renderParetoAplicacion(result){
  try{
    const pareto = result?.pareto || null;

    const titleEl = document.getElementById("inlineParetoTitle");
    const wrap = document.getElementById("paretoChartWrap");
    const canvas = document.getElementById("paretoChart");
    if(!titleEl || !wrap || !canvas) return;

    const items = pareto?.items || [];
    if(!Array.isArray(items) || items.length === 0){
      destroyPareto();
      return;
    }

    titleEl.textContent = "Análisis de Pareto de Causas de Parada";
    titleEl.style.display = "block";
    wrap.style.display = "block";

    const legendWrap = document.getElementById("paretoLegendWrap");
    const tbody = document.getElementById("paretoLegendBody");
    if(legendWrap && tbody){
      tbody.innerHTML = "";

      items.forEach(it=>{
        const label = String(it.label ?? "").trim();

        let code = "";
        let desc = label;

        const m = label.match(/^\s*(\d+)\s*-\s*(.+)\s*$/);
        if(m){
          code = m[1];
          desc = m[2];
        }else{
          const m2 = label.match(/^\s*(\d+)\s*$/);
          if(m2){
            code = m2[1];
            desc = STOP_CODE_DESC[code] || label;
          }else{
            code = "";
            desc = label;
          }
        }

        const dur = String(it.hhmm ?? secToHHMM(it.seconds ?? 0));
        const cum = (typeof it.cum_pct === "number") ? it.cum_pct : 0;

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="width:120px;">${escHtml(code)}</td>
          <td>${escHtml(desc)}</td>
          <td style="width:120px; text-align:right;"><b>${escHtml(dur)}</b></td>
          <td style="width:90px; text-align:right;"><b>${escHtml(cum.toFixed(1))}</b>%</td>
        `;
        tbody.appendChild(tr);
      });

      legendWrap.style.display = "block";
    }

    if(typeof Chart === "undefined"){
      wrap.style.display = "none";
      return;
    }

    _destroyParetoChartOnly();

    const labels = items.map(it => String(it.label ?? ""));
    const durSec = items.map(it => Number(it.seconds ?? 0) || 0);
    const durMin = durSec.map(s => s / 60.0);
    const cumPct = items.map(it => Number(it.cum_pct ?? 0) || 0);

    _paretoChart = new Chart(canvas.getContext("2d"), {
      data: {
        labels,
        datasets: [
          { type: "bar",  label: "Duración (min)", data: durMin, yAxisID: "y" },
          { type: "line", label: "Acumulado (%)",  data: cumPct, yAxisID: "y1", tension: 0.25, pointRadius: 2 }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: function(ctx){
                if(ctx.dataset && ctx.dataset.type === "bar"){
                  const s = durSec[ctx.dataIndex] ?? 0;
                  return "Duración: " + secToHHMM(s);
                }
                return "Acumulado: " + (ctx.parsed?.y ?? ctx.raw) + "%";
              }
            }
          }
        },
        scales: {
          y:  { beginAtZero: true, title: { display: true, text: "Duración (min)" } },
          y1: { beginAtZero: true, min: 0, max: 100, position: "right",
                grid: { drawOnChartArea: false },
                title: { display: true, text: "Acumulado (%)" } }
        }
      }
    });

  }catch(e){
    console.error("renderParetoAplicacion error:", e);
  }
}

// =========================
// Productividad por hora (SOLO DAY)
// =========================
function hideProdHour(){
  const row = document.getElementById("prodHourTitleRow");
  if(row) row.style.display = "none";

  const t = document.getElementById("prodHourTitle");
  const w = document.getElementById("prodHourWrap");
  const g = document.getElementById("prodHourGrid");
  const info = document.getElementById("lenRangesInfo");
  const leg = document.getElementById("prodHourLegend");

  if(g) g.innerHTML = "";
  if(t) t.style.display = "none";
  if(w) w.style.display = "none";
  if(info) info.style.display = "none";
  if(leg) leg.style.display = "none";
}



function hourRangeLabel(hourLabel){
  const raw = String(hourLabel || "").trim();
  if(!raw) return "—";
  if(raw.includes("-")) return raw;

  const h = parseInt(raw.split(":")[0], 10);
  if(Number.isNaN(h)) return raw;

  const h2 = (h + 1) % 24;
  const a = String(h).padStart(2,"0") + ":00";
  const b = String(h2).padStart(2,"0") + ":00";
  return `${a} - ${b}`;
}

function _causeSec(c){
  if(!c) return 0;
  return Number(c.seconds ?? c.sec ?? c.value ?? 0) || 0;
}
function _causeLabel(c){
  return String(c.label ?? c.cause ?? c.name ?? "").trim();
}



function renderProdHour(result){
  const period = String(result?.period || "");
  const title = document.getElementById("prodHourTitle");
  const wrap  = document.getElementById("prodHourWrap");
  const grid  = document.getElementById("prodHourGrid");
  const info  = document.getElementById("lenRangesInfo");
  if(!title || !wrap || !grid) return;

  if(period !== "day"){
    hideProdHour();
    return;
  }

  const buckets = result?.kpis?.hourly_buckets || [];
  if(!Array.isArray(buckets) || buckets.length === 0){
    hideProdHour();
    return;
  }

  const machineU = String(result?.machine || "").toUpperCase();
  const isAplic = (machineU === "APLICACION" || machineU === "UNION");
  const showLenMix = (!isAplic) && (machineU !== "HP");
  const showMeters = (!isAplic) && (machineU !== "HP");

  grid.innerHTML = "";

  buckets.forEach(b=>{
    const hourStart = String(b.hourLabel || b.hour || "").trim() || "";
    const hour = hourRangeLabel(hourStart);

    // ✅ THB: calcular inicio/fin hora para consulta de tarjetas
    const hsClean = (hourStart.includes("-") ? hourStart.split("-")[0].trim() : hourStart);
    const heClean = (hourStart.includes("-") ? (hourStart.split("-")[1] || "").trim() : "");

    const hourEnd = (() => {
      if(heClean) return heClean;
      const raw = String(hsClean || "").trim();
      const h = parseInt(raw.split(":")[0], 10);
      if(Number.isNaN(h)) return "";
      const h2 = (h + 1) % 24;
      return String(h2).padStart(2,"0") + ":00";
    })();

    const circuitsAttrs = (machineU === "THB")
      ? ` data-hour-start="${escHtml(hsClean)}" data-hour-end="${escHtml(hourEnd)}"
          tabindex="0" role="button"
          title="Ver tarjetas (Consecutivo / Código del arnés)"`
      : ``;

    const circuits = Number(b.cut)||0;
    const meters = Number(b.meters)||0;

    // ======= ✅ NORMALIZACIÓN DE UNIDADES (HORAS DECIMALES -> SEGUNDOS) =======
    let prodSec = Number(b.prodSec)||0;
    let deadSec = Number(b.deadSec)||0;
    let mealSec = Number(b.mealSec)||0;

    // si vienen como horas decimales (ej 0.36, 0.63, 0.26), convierto a segundos
    const looksLikeHours =
      (prodSec > 0 && prodSec < 24) || (deadSec > 0 && deadSec < 24) || (mealSec > 0 && mealSec < 24);

    // más estricto: si hay decimales y valores pequeños, casi seguro son horas
    const hasDecimals = (x)=> Math.abs(x - Math.trunc(x)) > 1e-9;
    const probablyDecimalHours =
      (hasDecimals(prodSec) && prodSec < 10) ||
      (hasDecimals(deadSec) && deadSec < 10) ||
      (hasDecimals(mealSec) && mealSec < 10);

    const toSeconds = (v)=> Math.round(v * 3600);

    if(looksLikeHours && probablyDecimalHours){
      prodSec = toSeconds(prodSec);
      deadSec = toSeconds(deadSec);
      mealSec = toSeconds(mealSec);
    }
    // ======================================================================

    // ✅ “Otro” = SOLO lo real del backend. NO completar a 3600.
    let otherDeadSecFinal = (b.otherDeadSec != null)
      ? (Number(b.otherDeadSec)||0)
      : Math.max(0, deadSec - mealSec);

    // si otherDeadSec también viene como horas decimales, convierto igual
    if(looksLikeHours && probablyDecimalHours && b.otherDeadSec != null){
      otherDeadSecFinal = toSeconds(Number(b.otherDeadSec)||0);
    }

    if(machineU !== "HP"){
      const sumHour = Math.max(0, prodSec) + Math.max(0, otherDeadSecFinal) + Math.max(0, mealSec);
      const delta = 3600 - sumHour;
      if(delta === 60){
        prodSec += 60;
      }
    }




    // ✅ % sobre el MISMO total que muestran los decimales (prod + other + comida)
    // ✅ Porcentajes: cerrados a 100% usando (prod + other + comida)
    const pct = _fmtPctPartsTo100(prodSec, otherDeadSecFinal, mealSec);
    // ✅ pct.prodPct viene como "52%" -> número 52
    const pctEffN = parseInt(String(pct.prodPct || "0").replace("%",""), 10) || 0;





    const isZero = circuits <= 0;
    const good = (!isZero) && (pctEffN >= 50);

    const mix = b.len_mix || null;
    const smallPct = mix ? Math.round(Number(mix.short_pct)||0) : 0;
    const medPct   = mix ? Math.round(Number(mix.mid_pct)||0)   : 0;
    const longPct  = mix ? Math.round(Number(mix.long_pct)||0)  : 0;
    const hasMix = mix && ((smallPct + medPct + longPct) > 0);

    let sPct = smallPct, mPct = medPct, lPct = longPct;
    sPct = Math.max(0, Math.min(100, sPct));
    mPct = Math.max(0, Math.min(100, mPct));
    lPct = Math.max(0, Math.min(100, lPct));
    const sum = sPct + mPct + lPct;
    if(sum !== 100){
      lPct = Math.max(0, 100 - sPct - mPct);
    }

    const mixHtml = showLenMix ? (
      hasMix ? `
        <div class="len-mix">
          <p class="len-title">Longitudes del cable</p>
          <div class="len-bar" title="Pequeño ${sPct}% · Mediano ${mPct}% · Largo ${lPct}%">
            <div class="len-seg short" style="width:${sPct}%">${sPct}%</div>
            <div class="len-seg mid"   style="width:${mPct}%">${mPct}%</div>
            <div class="len-seg long"  style="width:${lPct}%">${lPct}%</div>
          </div>
        </div>
      ` : `
        <div class="len-mix">
          <p class="len-title">Longitudes del cable</p>
          <div class="len-bar">
            <div class="len-seg short" style="width:100%">Sin datos</div>
          </div>
        </div>
      `
    ) : ``;

    let otherCauses = Array.isArray(b.other_causes) ? b.other_causes.slice() : [];

    function buildOtherPanelHtml(){
      function normOtherKey(x){
        return String(x || "")
          .trim()
          .toLowerCase()
          .replace(/\s+/g, "")
          .replace(/_+/g, "_")
          .replace(/_?\-+\>_?/g, "→")
          .replace(/\-_+/g, "→")
          .replace(/_?→_?/g, "→")
          .replace(/_/g, "");
      }

      function prettifyFallback(s){
        return String(s || "")
          .replace(/\s+/g, " ")
          .replace(/_+/g, " ")
          .replace(/\s*-\s*>\s*/g, " → ")
          .trim();
      }

      if(!otherCauses.length){
        return `<div class="ph-otro-panel"><div class="ph-otro-title">Causas de Paro</div>
                <div class="ph-otro-muted">Sin detalle de causas para esta hora.</div></div>`;
      }

      const agg = new Map();
      let _seq203 = 0; // ✅ contador para NO agrupar 203

      for(const it of otherCauses){
        const sec = Number(it?.seconds ?? it?.sec ?? it?.downtime_s ?? it?.value ?? 0) || 0;
        if(sec <= 0) continue;

        const rawLabel = String(it?.label ?? it?.pareto_label ?? it?.key ?? "").trim();
        let code = String(it?.code ?? it?.cause_code ?? it?.CausaCode ?? it?.causa ?? "").trim();
        let desc = String(it?.desc ?? it?.description ?? it?.name ?? "").trim();

        if(!code && rawLabel){
          const m = rawLabel.match(/^\s*(\d+)\s*-\s*(.+)\s*$/);
          if(m){
            code = m[1];
            if(!desc) desc = m[2];
          }else{
            const m2 = rawLabel.match(/^\s*(\d+)\s*$/);
            if(m2) code = m2[1];
          }
        }

        let kind = "other";
        let key = "";

        if(/^\d+$/.test(code)){
          kind = "stopcode";
          desc = STOP_CODE_DESC[code] || desc || "—";

            // ✅ 203 NO se agrupa: cada ocurrencia queda como item separado
          if(String(code) === "203"){
            _seq203 += 1;
            key = `code:203#${_seq203}`;   // clave única => no suma
          }else{
            key = `code:${code}`;         // otros códigos sí se agrupan normal
          }

        }else{
          const norm = normOtherKey(rawLabel || code || desc);
          key = `other:${norm || "unknown"}`;
          const friendly = OTHER_INTERVAL_DESC[norm] || "";
          desc = friendly || desc || prettifyFallback(rawLabel || code) || "—";
          code = "";
        }

        const prev = agg.get(key);
        if(prev) prev.seconds += sec;
        else agg.set(key, { kind, code, desc, seconds: sec });
      }


      const arr = Array.from(agg.values()).sort((a,b)=> (b.seconds||0)-(a.seconds||0));
      if(!arr.length){
        return `<div class="ph-otro-panel"><div class="ph-otro-title">Causas de Paro</div>
                <div class="ph-otro-muted">Sin detalle de causas para esta hora.</div></div>`;
      }

            // ====== ✅ TOP 12 + FALTANTE A 000 (para cuadrar con el BADGE "Otro") ======
      // ✅ HP: NO cuadrar con 000. THB: sí puede cuadrar como antes.
      const top = arr.slice(0, 12);

      if(machineU !== "HP"){
      // Lo que muestra el badge "Otro" (ya viene redondeado a 2 decimales por fmtParts)
        const badgeOtherDec = Number(fmtParts?.otherStr || "0") || 0;

      // Sumar lo que se verá en la lista (mismo truncado que usas en secToHHMMSS)
        const decFromSec = (sec)=> Number(_fmtHourDecFromSec(sec)) || 0;

        let sumShownDec = 0;
        for(const it of top){
          sumShownDec += decFromSec(it.seconds || 0);
        }

        const missingDec = _trunc2(Math.max(0, badgeOtherDec - sumShownDec));

        if(missingDec > 0){
          const missingSec = Math.round(missingDec * 3600);

        // Busca 000 en el top para sumarle; si no está, lo crea
          let idx000 = top.findIndex(x => String(x.code || "").trim() === "000");
          if(idx000 >= 0){
            top[idx000].seconds = (Number(top[idx000].seconds)||0) + missingSec;
            top[idx000].kind = "stopcode";
            top[idx000].desc = STOP_CODE_DESC["000"] || top[idx000].desc || "Desconocido";
          }else{
            top.push({ kind:"stopcode", code:"000", desc:(STOP_CODE_DESC["000"] || "Desconocido"), seconds: missingSec });
          }
        }
      }


      const items = top.map(it=>{
        const hhmm = secToHHMMSS(it.seconds || 0);
        if(it.kind === "stopcode" && it.code){
          return `<li><b>${escHtml(it.code)}</b> — ${escHtml(it.desc)} <span class="ph-otro-muted">(${escHtml(hhmm)})</span></li>`;
        }
        return `<li>${escHtml(it.desc)} <span class="ph-otro-muted">(${escHtml(hhmm)})</span></li>`;
      }).join("");
      // ======================================================================


      return `<div class="ph-otro-panel">
                <div class="ph-otro-title">Causas de Paro</div>
                <ol class="ph-otro-list">${items}</ol>
              </div>`;
    }
    // ✅ Cerrar decimales para que la SUMA sea igual al total real (prod + other + comida)
    const fmtParts = (machineU === "HP")
      ? _fmtHourDecPartsRaw(prodSec, otherDeadSecFinal, mealSec)   // ✅ HP: sin ajustar
      : _fmtHourDecPartsToTotal(prodSec, otherDeadSecFinal, mealSec); // ✅ THB/otros: igual como estaba


    const timeBadges = (mealSec > 0) ? `
      <div class="ph-badge good">
        <span>Efectivo: ${escHtml(fmtParts.prodStr)}</span>
        <span class="ph-badge-pct">${pct.prodPct}</span>
      </div>

      <button type="button" class="ph-badge bad ph-otro-btn">
        <span>Otro: ${escHtml(fmtParts.otherStr)}</span>
        <span class="ph-badge-pct">${pct.otherPct}</span>
      </button>

      <div class="ph-badge meal">
        <span>Comida: ${escHtml(fmtParts.mealStr)}</span>
        <span class="ph-badge-pct">${pct.mealPct}</span>
      </div>
    ` : `
      <div class="ph-badge good">
        <span>Efectivo: ${escHtml(fmtParts.prodStr)}</span>
        <span class="ph-badge-pct">${pct.prodPct}</span>
      </div>

      <button type="button" class="ph-badge bad ph-otro-btn">
        <span>Otro: ${escHtml(fmtParts.otherStr)}</span>
        <span class="ph-badge-pct">${pct.otherPct}</span>
      </button>
    `;

    const metersHtml = showMeters
      ? `<div class="ph-meters">${escHtml(fmtMetersEs(meters))}</div>`
      : ``;

    const card = document.createElement("div");
    card.className = "ph-card " + (isZero ? "ph-zero" : (good ? "ph-good" : "ph-bad"));
    card.innerHTML = `
      <div class="ph-hour">${escHtml(hour)}</div>
      <div class="ph-circuits ${(machineU === "THB") ? "ph-circuits-link" : ""}"${circuitsAttrs}>
        ${escHtml(fmtIntEs(circuits))}
      </div>
      <div class="ph-unit">${(machineU === "APLICACION" || machineU === "UNION") ? "Crimpados" : "Circuitos"}</div>
      ${metersHtml}

      <div class="ph-badges">
        ${timeBadges}
      </div>

      ${buildOtherPanelHtml()}
      ${mixHtml}
    `;
    grid.appendChild(card);

    const btnOtro = card.querySelector(".ph-otro-btn");
    const panel = card.querySelector(".ph-otro-panel");
    if(btnOtro && panel){
      btnOtro.addEventListener("click", (ev)=>{
        ev.stopPropagation();
        document.querySelectorAll("#prodHourGrid .ph-otro-panel.open").forEach(p=>{
          if(p !== panel) p.classList.remove("open");
        });
        panel.classList.toggle("open");
      });
    }

    const circuitsLink = card.querySelector(".ph-circuits-link");
    if(circuitsLink){
      const pvSel = document.getElementById("periodValueSelect");
      const opSel = document.getElementById("operatorSelect");

      const payload = {
        filename: String(CFG.filename || ""),
        period_value: String(pvSel ? pvSel.value : "").trim(),
        operator: String(opSel ? opSel.value : "").trim(),
        hour_start: String(circuitsLink.dataset.hourStart || hsClean || ""),
        hour_end: String(circuitsLink.dataset.hourEnd || hourEnd || ""),
        hour_label: String(hour || ""),
        circuits: circuits
      };

      circuitsLink.addEventListener("click", (ev)=>{
        ev.stopPropagation();
        showThbTarjetasSide(payload, circuitsLink);
      });

      circuitsLink.addEventListener("keydown", (ev)=>{
        if(ev.key === "Enter" || ev.key === " "){
          ev.preventDefault();
          circuitsLink.click();
        }
      });
    }
  });

  const row = document.getElementById("prodHourTitleRow");
  if(row) row.style.display = "flex";

  title.style.display = "block";
  wrap.style.display = "block";
  const leg = document.getElementById("prodHourLegend");
  if(leg) leg.style.display = "flex";

  if(info) info.style.display = showLenMix ? "block" : "none";
}








// =========================
// INLINE RESULTS (AJAX)
// =========================
function renderInlineKPIs(result){
  const box = document.getElementById("inlineResults");
  const hdr = document.getElementById("inlineHeader");
  const ttl = document.getElementById("inlineTitle");
  const kpi = document.getElementById("inlineKpis");
  const paretoTtl = document.getElementById("inlineParetoTitle");
  const dbg = document.getElementById("inlineDebug");
  const dbgPre = document.getElementById("inlineDebugPre");

  if(!box || !hdr || !ttl || !kpi || !paretoTtl || !dbg || !dbgPre) return;

  box.style.display = "block";
  dbg.style.display = "none";
  dbgPre.textContent = "";
  kpi.innerHTML = "";

  ttl.style.display = "none";
  paretoTtl.style.display = "none";
  paretoTtl.textContent = "";

  destroyPareto();
  hideProdHour();

  hdr.innerHTML = `
    Máquina: <b>${escHtml(result.machine||"")}</b>
    ${result.period && result.period_value ? `— Periodo: <b>${escHtml(result.period)}</b> (<b>${escHtml(result.period_value)}</b>)` : ""}
    ${result.operator ? `— Operaria: <b>${escHtml(result.operator)}</b>` : ""}
    ${result.time_start && result.time_end ? `— Hora: <b>${escHtml(result.time_start)}</b> a <b>${escHtml(result.time_end)}</b>` : ""}
    ${result.rows_total != null ? `— Filas: <b>${escHtml(result.rows_total)}</b>` : ""}
  `;

  ttl.textContent = periodTitle(result.period);
  ttl.style.display = "block";

  const ui = result?.kpis_ui;
  if(!ui || typeof ui !== "object"){
    dbg.style.display = "block";
    dbgPre.textContent = JSON.stringify(result, null, 2);
    return;
  }
  const m = String(result.machine || "").toUpperCase();

  let order = [
    { key: "Circuitos Cortados",            cls: "kpi-green",  span: "kpi-row-3" },
    { key: "Circuitos Planeados",           cls: "kpi-cyan",   span: "kpi-row-3" },
    { key: "Circuitos No Conformes",        cls: "kpi-red",    span: "kpi-row-3" },

    { key: "Tiempos Perdidos ", cls: "kpi-blue",   span: "kpi-row-2" },
    { key: "Tiempo Trabajado",               cls: "kpi-green2", span: "kpi-row-2" },

    { key: "Metros Planeados",              cls: "kpi-purple", span: "kpi-row-3" },
    { key: "Metros Extras",                 cls: "kpi-blue2",  span: "kpi-row-3" },
    { key: "Metros Cortados",               cls: "kpi-cyan2",  span: "kpi-row-3" },
  ];

  // ✅ SOLO THB: poner Horas/Hombre en la fila de tiempos, al lado de "Otro Tiempo"
    if(m === "THB"){
      order = [
        { key: "Circuitos Cortados",     cls: "kpi-green",  span: "kpi-row-3" },
        { key: "Circuitos Planeados",    cls: "kpi-cyan",   span: "kpi-row-3" },
        { key: "Circuitos No Conformes", cls: "kpi-red",    span: "kpi-row-3" },

        { key: "Horas Disponibles",                  cls: "",           span: "kpi-row-4" }, // 3 cols (al lado)
        { key: "Tiempos Perdidos", cls: "kpi-blue",   span: "kpi-row-2" }, // 6 cols
        { key: "Tiempo Trabajado",               cls: "kpi-green2", span: "kpi-row-4" }, // 3 cols

        { key: "Metros Planeados", cls: "kpi-purple", span: "kpi-row-3" },
        { key: "Metros Extras",    cls: "kpi-blue2",  span: "kpi-row-3" },
        { key: "Metros Cortados",  cls: "kpi-cyan2",  span: "kpi-row-3" },
      ];
    }

      // ✅ SOLO HP: Horas Disponibles en la fila de tiempos
  if(m === "HP"){
    order = [
      { key: "Circuitos Cortados",     cls: "kpi-green",  span: "kpi-row-3" },
      { key: "Circuitos Planeados",    cls: "kpi-cyan",   span: "kpi-row-3" },
      { key: "Circuitos No Conformes", cls: "kpi-red",    span: "kpi-row-3" },

      { key: "Horas Disponibles",            cls: "",           span: "kpi-row-4" },
      { key: "Tiempos Perdidos", cls: "kpi-blue",   span: "kpi-row-2" },
      { key: "Tiempo Trabajado",               cls: "kpi-green2", span: "kpi-row-4" },
    ];
  }





    if(m === "APLICACION" || m === "UNION"){
      order = [
        { key: "Crimpados",         cls: "kpi-green", span: "kpi-row-3" },
        { key: "Crimpados Carrete", cls: "kpi-cyan",  span: "kpi-row-3" },
        { key: "Crimpados Manual",  cls: "kpi-red",   span: "kpi-row-3" },

        // ✅ Fila de tiempos (como THB/HP): izquierda 3 cols, centro 6 cols, derecha 3 cols
        { key: "Horas activas",                 cls: "",          span: "kpi-row-4" }, // ✅ arriba izq
        { key: "Tiempos Perdidos", cls: "kpi-purple",span: "kpi-row-2" }, // ✅ grande centro
        { key: "Tiempo Trabajado",               cls: "kpi-green2",span: "kpi-row-4" }, // ✅ arriba der

        { key: "Paradas planeadas (HH:MM)",    cls: "kpi-cyan2", span: "kpi-row-2" },
        { key: "Paradas no planeadas (HH:MM)", cls: "kpi-blue2", span: "kpi-row-2" },
      ];
    }


  order.forEach(item => {
    if(Object.prototype.hasOwnProperty.call(ui, item.key)){
      kpi.appendChild(makeCard(item.key, ui[item.key], item.cls, item.span));
    }
  });

  for(const [kk, vv] of Object.entries(ui)){
    if(!order.some(o => o.key === kk)){
      kpi.appendChild(makeCard(kk, vv, "", ""));
    }
  }

    // ✅ HP: forzar cierre de No registrado con Horas Disponibles
  if(m === "HP" || m === "THB"){
    adjustHpNoRegistradoUI();
  }


  renderHpTimesLine(result);

  const mu = String(result.machine || "").toUpperCase();
  if(mu === "APLICACION" || mu === "UNION"){
    if(result?.pareto?.items?.length){
      paretoTtl.style.display = "block";
      renderParetoAplicacion(result);
    }else{
      paretoTtl.style.display = "none";
      destroyPareto();
    }
  }else{
    if(result?.pareto?.labels?.length){
      paretoTtl.textContent = paretoTitle(String(result.period || ""));
      paretoTtl.style.display = "block";
      renderParetoParadas(result);
    }else{
      paretoTtl.style.display = "none";
      destroyPareto();
    }
  }

  renderTerminalUsage(result);
  renderProdHour(result);
}

async function runInlineAnalysis(form, opts = {}){
  const btn = document.getElementById("btnAnalyze");
  if(btn){ btn.disabled = true; btn.textContent = "Cargando..."; }

  const doScroll = (opts.scroll !== undefined) ? !!opts.scroll : true;
  const preserveScroll = !!opts.preserveScroll;

  // ✅ Captura ancla SOLO si es refresh automático (preserveScroll)
  const anchor = preserveScroll ? _captureHourAnchor() : null;

  if(!periodValueOk()){
    if(btn){ btn.disabled = false; btn.textContent = "Ver KPIs"; }
    alert("Selecciona un Día/Semana/Mes antes de ejecutar el análisis.");
    return;
  }

  try{
    const fd = new FormData(form);

    // ✅ garantizar que machine_id viaje al backend (select_sheet)
    if(!fd.has("machine_id")){
      const mid = (document.getElementById("machineIdInput2")?.value || _cfgStr(CFG.machine_id) || "").trim();
      if(mid) fd.append("machine_id", mid);
    }

    // ✅ garantizar que machine viaje como el hidden real
    if(fd.has("machine")){
      fd.set("machine", getCurrentMachine());
    }else{
      fd.append("machine", getCurrentMachine());
    }

    if(!URLS.run_analysis_api){
      alert("Error: URL run_analysis_api no configurada.");
      return;
    }

    const url = `${URLS.run_analysis_api}?_=${Date.now()}`;
    const r = await fetch(url, {
      method: "POST",
      body: fd,
      cache: "no-store"
    });

    const data = await r.json();

    if(!data.ok){
      alert(data.error || "Error ejecutando análisis.");
      return;
    }

    renderInlineKPIs(data.result);

    // ✅ Scroll control
    if(doScroll){
      document.getElementById("inlineResults")
        ?.scrollIntoView({ behavior:"smooth", block:"start" });
    }else if(preserveScroll){
      // Esperar reflow y restaurar
      requestAnimationFrame(()=>requestAnimationFrame(()=>_restoreHourAnchor(anchor)));
    }

  }catch(e){
    alert("Error de red / servidor ejecutando el análisis.");
  }finally{
    if(btn){ btn.disabled = false; btn.textContent = "Ver KPIs"; }
  }
}


function setupCicloAplicadores(){
  // ✅ Fuerza machine a APLICACION (por si acaso)
  const machineInput = document.getElementById("machineInput");
  const machineLabel = document.getElementById("machineLabel");
  if(machineInput) machineInput.value = "APLICACION";
  if(machineLabel) machineLabel.textContent = "APLICACION";

  // ✅ Ocultar cards de periodo (día/semana/mes)
  document.querySelectorAll(".period-card").forEach(c => {
    c.style.display = "none";
  });

  // ✅ Ocultar dropdown de period_value (y poner un valor dummy)
  const pvSel = document.getElementById("periodValueSelect");
  if(pvSel){
    pvSel.innerHTML = `<option value="TOTAL">TOTAL</option>`;
    pvSel.value = "TOTAL";

    // intenta ocultar contenedor sin romper si cambia el HTML
    const box = pvSel.closest(".select-box") || pvSel.parentElement;
    if(box) box.style.display = "none";
  }

  // ✅ Label del periodo
  const lbl = document.getElementById("periodLabel");
  if(lbl) lbl.textContent = "Total histórico (4 máquinas)";

  // ✅ periodInput no importa, pero dejamos “day” para consistencia
  const periodInput = document.getElementById("periodInput");
  if(periodInput) periodInput.value = "day";

  // ✅ Operador fijo en General
  const opSel = document.getElementById("operatorSelect");
  if(opSel){
    opSel.innerHTML = `<option value="General">General</option>`;
    opSel.value = "General";
  }

  // ✅ No horas
  setTimeApply(false);
  setTimeInputsVisible(false);

  // ✅ Muestra bloque de filtros + habilita botón
  const box = document.getElementById("thbFilters");
  const btn = document.getElementById("btnAnalyze");
  if(box) box.style.display = "block";
  if(btn) btn.disabled = false;

  // ✅ Auto-run apenas carga la vista
  const form = document.getElementById("runForm");
  if(form){
    // pequeño delay para que la UI pinte primero
    setTimeout(()=> runInlineAnalysis(form, { scroll:true }), 200);
  }
}


// =========================
// DOM Ready
// =========================
document.addEventListener("DOMContentLoaded", ()=>{

  const form = document.getElementById("runForm");
  if(form){
    form.addEventListener("submit", (ev)=>{
      ev.preventDefault();
      runInlineAnalysis(form, { scroll:true });
    });
  }

  // Inicial (por si estás en upload)
  gateUploadBlock();

  // Si estás en select_sheet, carga opciones
  // Si estás en select_sheet, carga opciones (EXCEPTO ciclo/aplicadores)
  if(_cfgStr(CFG.stage) === "select_sheet"){
    if(__IS_CICLO_LINE){
      setupCicloAplicadores();   // ✅ corre directo
    }else{
      reloadOptions();
    }
  }


    // =========================
    // LIVE REFRESH (Aplicación/Unión + Corte) — SOLO HOY con registros
    // =========================
    const AUTO_REFRESH_MS = 30000;
    let __autoTimer = null;
    let __inFlight = false;

    function __isLiveRefreshContext(){
      const m = getCurrentMachineU(); // usa hidden real
      const lk = _cfgStr(CFG.line_key);

      // Aplicación/Unión (igual que antes)
      if((lk === "aplicacion") || (m === "APLICACION") || (m === "UNION")) return true;

      // Corte (nuevo): solo HP/THB
      if((lk === "corte") && (m === "HP" || m === "THB")) return true;

      return false;
    }

    function __todayISO(){
      const d = new Date();
      const y = d.getFullYear();
      const m = String(d.getMonth()+1).padStart(2,"0");
      const dd = String(d.getDate()).padStart(2,"0");
      return `${y}-${m}-${dd}`; // 2026-01-30
    }

    function __selectedIsToday(){
      const pv = (document.getElementById("periodValueSelect")?.value || "").trim();
      if(!pv) return false;

      // Esperamos value ISO (yyyy-mm-dd). Si viene con '/', lo normalizamos.
      const norm = pv.replaceAll("/", "-");
      return norm === __todayISO();
    }

    function __todayExistsInOptions(){
      const sel = document.getElementById("periodValueSelect");
      if(!sel) return false;

      const t = __todayISO();
      const opts = Array.from(sel.options || []);

      // ✅ comparar SOLO contra value (ISO) para que el label pueda ser DD/MM/YYYY sin romper
      return opts.some(op => String(op.value || "").trim().replaceAll("/", "-") === t);
    }

    function __canAutoRefreshToday(){
      if(document.hidden) return false;
      if(!__isLiveRefreshContext()) return false;

      const period = (document.getElementById("periodInput")?.value || "day").trim();
      if(period !== "day") return false;          // 🔒 semanas/mes NO refrescan

      if(!__selectedIsToday()) return false;      // 🔒 días pasados NO refrescan
      if(!__todayExistsInOptions()) return false; // 🔒 si HOY no tiene registros, NO refresca

      return true;
    }

    async function __refreshTick(){
      if(__inFlight) return;
      if(!__canAutoRefreshToday()) return;

      const f = document.getElementById("runForm");
      if(!f) return;

      __inFlight = true;
      try{
        await runInlineAnalysis(f, { scroll:false, preserveScroll:true });
      }catch(e){
        console.warn("Auto-refresh fallo:", e);
      }finally{
        __inFlight = false;
      }
    }

    function startAutoRefresh(){
      if(__autoTimer) return;
      __autoTimer = setInterval(__refreshTick, AUTO_REFRESH_MS);
    }

    function stopAutoRefresh(){
      if(!__autoTimer) return;
      clearInterval(__autoTimer);
      __autoTimer = null;
    }

    // Si cambias day/week/month o el valor, intenta tick (pero solo hará algo si es HOY)
    document.querySelectorAll(".period-card").forEach(card=>{
      card.addEventListener("click", ()=> setTimeout(__refreshTick, 800));
    });
    document.getElementById("periodValueSelect")?.addEventListener("change", ()=> __refreshTick());

    document.addEventListener("visibilitychange", ()=>{
      if(document.hidden) stopAutoRefresh();
      else startAutoRefresh();
    });

    startAutoRefresh();

});

// =========================
// THB: Modal Tarjetas por hora (Consecutivo / Código del arnés)
// =========================
function _thbModalEls(){
  return {
    modal: document.getElementById("thbTarjetasModal"),
    close: document.getElementById("thbTarjetasClose"),
    sub: document.getElementById("thbTarjetasSub"),
    body: document.getElementById("thbTarjetasBody"),
  };
}

function _thbModalOpen(){
  const { modal, close } = _thbModalEls();
  if(!modal) return;

  modal.style.display = "block";

  // backdrop click
  const bd = modal.querySelector("[data-close='1']");
  if(bd){
    bd.onclick = ()=> _thbModalClose();
  }
  if(close){
    close.onclick = ()=> _thbModalClose();
  }

  // ESC
  document.addEventListener("keydown", _thbModalEscOnce, { once: true });
}
function _thbModalEscOnce(ev){
  if(ev.key === "Escape") _thbModalClose();
  else document.addEventListener("keydown", _thbModalEscOnce, { once: true });
}

function _thbModalClose(){
  const { modal } = _thbModalEls();
  if(!modal) return;
  modal.style.display = "none";
}

function _positionThbModalNear(anchorEl){
  const modal = document.getElementById("thbTarjetasModal");
  if(!modal) return;

  const card =
    modal.querySelector(".thb-modal-card") ||
    modal.querySelector(".modal-card") ||
    modal.querySelector(".thb-card") ||
    modal.firstElementChild;

  if(!card) return;

  // ✅ anclar a la TARJETA completa (ph-card), no al numerito
  const host = (anchorEl && anchorEl.closest)
    ? (anchorEl.closest(".ph-card") || anchorEl)
    : anchorEl;

  const r = (host && host.getBoundingClientRect)
    ? host.getBoundingClientRect()
    : (document.getElementById("prodHourWrap")?.getBoundingClientRect() || null);

  // ✅ TOP donde comienza la tarjeta (queda ENCIMA de las tarjetas hora/hora)
  const top = r ? Math.round(r.top + 8) : 1000;

  // overlay como flex y bajado con padding-top
  modal.style.setProperty("display", "flex", "important");
  modal.style.setProperty("justify-content", "center", "important");
  modal.style.setProperty("align-items", "flex-start", "important");
  modal.style.setProperty("padding-top", `${Math.max(40, top)}px`, "important");

  // card sin centrar raro por transform/top 50%
  card.style.setProperty("position", "relative", "important");
  card.style.setProperty("top", "0px", "important");
  card.style.setProperty("left", "0px", "important");
  card.style.setProperty("right", "auto", "important");
  card.style.setProperty("transform", "none", "important");
  card.style.setProperty("margin", "0 auto", "important");

  // body con scroll interno si no cabe
  const gap = 12;
  const headH = card.querySelector(".thb-modal-head")?.offsetHeight || 60;
  const body = card.querySelector(".thb-modal-body");
  if(body){
    const avail = window.innerHeight - Math.max(40, top) - gap - headH - 16;
    body.style.maxHeight = `${Math.max(180, Math.floor(avail))}px`;
    body.style.overflowY = "auto";
  }
}


// =========================
// THB: Popover al lado (mismo tamaño de la tarjeta)
// Cierra al click afuera
// =========================
let __thbSide = { open:false, anchor:null };

function _thbSideEnsure(){
  let pop = document.getElementById("thbTarjetasSidePop");
  if(pop) return pop;

  pop = document.createElement("div");
  pop.id = "thbTarjetasSidePop";

  // ✅ (Ítem 2-A) Clase para que agarre TODO el CSS bonito
  pop.className = "thb-side-pop";

  // ✅ estilos mínimos (lo visual lo maneja el CSS)
  pop.style.position = "fixed";
  pop.style.zIndex = "9999";
  pop.style.display = "none";
  pop.style.overflow = "hidden";

  // ✅ (Ítem 2-B) Estructura sin estilos inline (CSS manda)
  pop.innerHTML = `
    <div class="thb-side-head">
      <div class="thb-side-title"></div>
      <button type="button" class="thb-side-close" aria-label="Cerrar">×</button>
    </div>
    <div class="thb-side-body"></div>
  `;

  document.body.appendChild(pop);

  pop.querySelector(".thb-side-close")?.addEventListener("click", (ev)=>{
    ev.stopPropagation();
    _thbSideClose();
  });

  // Evita que un click dentro lo cierre
  pop.addEventListener("mousedown", (ev)=> ev.stopPropagation());
  pop.addEventListener("click", (ev)=> ev.stopPropagation());

  return pop;
}


function _thbSideClose(){
  const pop = document.getElementById("thbTarjetasSidePop");
  if(pop) pop.style.display = "none";
  __thbSide.open = false;
  __thbSide.anchor = null;
}

function _thbSidePosition(anchorEl){
  const pop = _thbSideEnsure();
  if(!anchorEl || !anchorEl.getBoundingClientRect) return;

  // tomamos la tarjeta completa (tamaño igual)
  const card = anchorEl.closest(".ph-card") || anchorEl;
  const r = card.getBoundingClientRect();

  const gap = 12;
  const w = Math.round(r.width);
  const h = Math.round(r.height);

  // tamaño igual a la tarjeta
  pop.style.width = `${w}px`;
  pop.style.height = `${h}px`;

  // posición: a la derecha por defecto; si no cabe, a la izquierda
  let left = Math.round(r.right + gap);
  const wouldOverflowRight = (left + w) > (window.innerWidth - 8);
  if(wouldOverflowRight){
    left = Math.round(r.left - gap - w);
  }

  // clamp horizontal
  left = Math.max(8, Math.min(left, window.innerWidth - w - 8));

  // top alineado con la tarjeta
  let top = Math.round(r.top);

  // clamp vertical (por si está muy arriba/abajo)
  top = Math.max(8, Math.min(top, window.innerHeight - h - 8));

  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;

  // body: que no se salga (scroll interno)
  const headH = pop.querySelector(".thb-side-head")?.offsetHeight || 44;
  const body = pop.querySelector(".thb-side-body");
  if(body){
    body.style.maxHeight = `${Math.max(120, h - headH)}px`;
  }
}

function _thbSideSetTitle(html){
  const pop = _thbSideEnsure();
  const t = pop.querySelector(".thb-side-title");
  if(t) t.innerHTML = html || "";
}


function _thbSideSetBody(html){
  const pop = _thbSideEnsure();
  const b = pop.querySelector(".thb-side-body");
  if(b) b.innerHTML = html;
}

function _thbSideOpen(anchorEl){
  const pop = _thbSideEnsure();
  pop.style.display = "block";
  __thbSide.open = true;
  __thbSide.anchor = anchorEl || null;
  _thbSidePosition(anchorEl);
}

// ✅ Listener global (una sola vez)
(function _thbSideBindOnce(){
  if(window.__THB_SIDE_BOUND) return;
  window.__THB_SIDE_BOUND = true;

  // click afuera => cerrar
  document.addEventListener("mousedown", (ev)=>{
    if(!__thbSide.open) return;

    const pop = document.getElementById("thbTarjetasSidePop");
    if(pop && pop.contains(ev.target)) return;

    // si clickeó otra vez el anchor actual, no cierres aquí (lo maneja el handler)
    if(__thbSide.anchor && ev.target === __thbSide.anchor) return;

    _thbSideClose();
  });

  // ESC => cerrar
  document.addEventListener("keydown", (ev)=>{
    if(ev.key === "Escape" && __thbSide.open) _thbSideClose();
  });

  // scroll/resize => reubicar
  window.addEventListener("scroll", ()=>{
    if(__thbSide.open) _thbSidePosition(__thbSide.anchor);
  }, { passive:true });

  window.addEventListener("resize", ()=>{
    if(__thbSide.open) _thbSidePosition(__thbSide.anchor);
  });
})();

async function showThbTarjetasSide(
  { filename, period_value, operator, hour_start, hour_end, hour_label, circuits },
  anchorEl
){
  const url = (URLS && URLS.thb_tarjetas_hour) ? String(URLS.thb_tarjetas_hour) : "";
  if(!url) return;

  // ✅ toggle: si vuelven a clickear el mismo numerito => cierra
  if(__thbSide.open && __thbSide.anchor === anchorEl){
    _thbSideClose();
    return;
  }

  // cerrar si había otro abierto
  _thbSideClose();

  // ✅ título 2 líneas (usa .muted del CSS)
  _thbSideSetTitle(
    `${escHtml(period_value || "—")} · ${escHtml(hour_label || (hour_start + " - " + hour_end))}` +
    `<span class="muted">Circuitos: ${escHtml(fmtIntEs(Number(circuits)||0))}</span>`
  );

  _thbSideSetBody(`<div style="opacity:.75;font-weight:800;">Cargando…</div>`);
  _thbSideOpen(anchorEl);

  try{
    const qs = new URLSearchParams({
      filename: String(filename || ""),
      period_value: String(period_value || ""),
      operator: String(operator || ""),
      hour_start: String(hour_start || ""),
      hour_end: String(hour_end || ""),
      limit: "250",
      _: String(Date.now())
    });

    const resp = await fetch(`${url}?${qs.toString()}`, { cache: "no-store" });
    const data = await resp.json();

    if(!resp.ok || !data || data.ok !== true){
      const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
      _thbSideSetBody(`<div style="opacity:.75;font-weight:800;">Error: ${escHtml(msg)}</div>`);
      return;
    }

    const items = Array.isArray(data.items) ? data.items : [];
    const total = Number(data.total || items.length || 0) || 0;
    const truncated = !!data.truncated;

    if(items.length === 0){
      _thbSideSetBody(`<div style="opacity:.75;font-weight:800;">Sin registros para esta hora.</div>`);
      return;
    }

    const rows = items.map((it, idx)=>`
      <tr>
        <td class="thb-idx">${idx+1}</td>
        <td class="thb-cons">${escHtml(it.consecutivo || "—")}</td>
        <td class="thb-code">${escHtml(it.codigo_arnes || "—")}</td>
      </tr>
    `).join("");

    _thbSideSetBody(`
      <table class="thb-side-table">
        <thead>
          <tr>
            <th class="thb-idx">#</th>
            <th>Consecutivo</th>
            <th>Código arnés</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>

      <div class="thb-side-footer">
        <span class="thb-pill">Total: ${escHtml(String(total))}</span>
        ${truncated ? `<span>Mostrando ${escHtml(String(items.length))}</span>` : `<span></span>`}
      </div>
    `);

    // reubicar por si cambió el alto al cargar
    requestAnimationFrame(()=> _thbSidePosition(anchorEl));

  }catch(e){
    _thbSideSetBody(`<div style="opacity:.75;font-weight:800;">Error: ${escHtml(String(e))}</div>`);
  }
}



async function showThbTarjetasByHour({ filename, period_value, operator, hour_start, hour_end, hour_label, circuits }, anchorEl){
  const { modal, sub, body } = _thbModalEls();
  if(!modal || !body) return;

  const url = (URLS && URLS.thb_tarjetas_hour) ? String(URLS.thb_tarjetas_hour) : "";
  if(!url){
    body.innerHTML = `<div class="thb-muted">Falta URLS.thb_tarjetas_hour en el CFG.</div>`;
    _thbModalOpen();
    return;
  }

  // header
  if(sub){
    sub.textContent = `${period_value || "—"} · ${hour_label || (hour_start + " - " + hour_end)} · Circuitos: ${fmtIntEs(Number(circuits)||0)}`;
  }

  body.innerHTML = `<div class="thb-modal-loading">Cargando…</div>`;
  _thbModalOpen();
  requestAnimationFrame(() => _positionThbModalNear(anchorEl)); // 👈 AQUÍ



  try{
    const qs = new URLSearchParams({
      filename: String(filename || ""),
      period_value: String(period_value || ""),
      operator: String(operator || ""),
      hour_start: String(hour_start || ""),
      hour_end: String(hour_end || ""),
      limit: "250",
      _: String(Date.now())
    });

    const resp = await fetch(`${url}?${qs.toString()}`, { cache: "no-store" });
    const data = await resp.json();

    if(!resp.ok || !data || data.ok !== true){
      const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
      body.innerHTML = `<div class="thb-muted">Error: ${escHtml(msg)}</div>`;

      return;
    }

    const items = Array.isArray(data.items) ? data.items : [];
    const total = Number(data.total || items.length || 0) || 0;
    const truncated = !!data.truncated;

    if(items.length === 0){
      body.innerHTML = `<div class="thb-muted">Sin registros para esta hora.</div>`;
      return;
    }

    const rows = items.map((it, idx)=>`
      <tr>
        <td style="width:60px; opacity:.7; font-weight:800;">${idx+1}</td>
        <td style="width:180px; font-weight:900;">${escHtml(it.consecutivo || "—")}</td>
        <td style="font-weight:800;">${escHtml(it.codigo_arnes || "—")}</td>
      </tr>
    `).join("");

    body.innerHTML = `
      <table class="thb-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Consecutivo</th>
            <th>Código del arnés</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="thb-muted">
        Total: <b>${escHtml(String(total))}</b>
        ${truncated ? ` · Mostrando primeros <b>${escHtml(String(items.length))}</b>` : ``}
      </div>
    `;

    requestAnimationFrame(() => _positionThbModalNear(anchorEl));

  }catch(e){
    body.innerHTML = `<div class="thb-muted">Error: ${escHtml(String(e))}</div>`;
  }
}

async function showThbTarjetasInline({ filename, period_value, operator, hour_start, hour_end, hour_label, circuits }, anchorEl){
  const grid = document.getElementById("prodHourGrid");
  if(!grid) return;

  const url = (URLS && URLS.thb_tarjetas_hour) ? String(URLS.thb_tarjetas_hour) : "";
  if(!url) return;

  // ✅ borrar panel previo si existe
  const old = document.getElementById("thbTarjetasInlinePanel");
  if(old) old.remove();

  // ✅ crear panel inline (spanea todo el grid)
  const panel = document.createElement("div");
  panel.id = "thbTarjetasInlinePanel";
  panel.style.gridColumn = "1 / -1";
  panel.style.background = "#fff";
  panel.style.borderRadius = "14px";
  panel.style.boxShadow = "0 10px 26px rgba(0,0,0,.10)";
  panel.style.border = "1px solid rgba(15,23,42,.10)";
  panel.style.padding = "12px 12px";
  panel.style.marginTop = "10px";

  panel.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px;">
      <div style="font-weight:900;">
        Tarjetas — ${escHtml(period_value || "—")} · ${escHtml(hour_label || (hour_start + " - " + hour_end))}
        · Circuitos: ${escHtml(fmtIntEs(Number(circuits)||0))}
      </div>
      <button type="button" class="thb-inline-close"
        style="border:0; background:#0f172a; color:#fff; padding:6px 10px; border-radius:10px; cursor:pointer; font-weight:800;">
        Cerrar
      </button>
    </div>

    <div class="thb-inline-body" style="max-height:360px; overflow:auto;">
      <div style="opacity:.7; font-weight:800;">Cargando…</div>
    </div>
  `;

  // ✅ insertarlo JUSTO debajo de la tarjeta clickeada (en el grid)
  const hostCard = (anchorEl && anchorEl.closest) ? anchorEl.closest(".ph-card") : null;
  if(hostCard && hostCard.parentElement === grid){
    grid.insertBefore(panel, hostCard.nextSibling);
  }else{
    grid.appendChild(panel);
  }

  panel.querySelector(".thb-inline-close")?.addEventListener("click", ()=> panel.remove());

  const body = panel.querySelector(".thb-inline-body");
  if(!body) return;

  try{
    const qs = new URLSearchParams({
      filename: String(filename || ""),
      period_value: String(period_value || ""),
      operator: String(operator || ""),
      hour_start: String(hour_start || ""),
      hour_end: String(hour_end || ""),
      limit: "250",
      _: String(Date.now())
    });

    const resp = await fetch(`${url}?${qs.toString()}`, { cache: "no-store" });
    const data = await resp.json();

    if(!resp.ok || !data || data.ok !== true){
      const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
      body.innerHTML = `<div style="opacity:.7; font-weight:800;">Error: ${escHtml(msg)}</div>`;
      return;
    }

    const items = Array.isArray(data.items) ? data.items : [];
    const total = Number(data.total || items.length || 0) || 0;
    const truncated = !!data.truncated;

    if(items.length === 0){
      body.innerHTML = `<div style="opacity:.7; font-weight:800;">Sin registros para esta hora.</div>`;
      return;
    }

    const rows = items.map((it, idx)=>`
      <tr>
        <td style="width:60px; opacity:.7; font-weight:800;">${idx+1}</td>
        <td style="width:180px; font-weight:900;">${escHtml(it.consecutivo || "—")}</td>
        <td style="font-weight:800;">${escHtml(it.codigo_arnes || "—")}</td>
      </tr>
    `).join("");

    body.innerHTML = `
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="text-align:left; padding:8px 6px; border-bottom:1px solid rgba(15,23,42,.12);">#</th>
            <th style="text-align:left; padding:8px 6px; border-bottom:1px solid rgba(15,23,42,.12);">Consecutivo</th>
            <th style="text-align:left; padding:8px 6px; border-bottom:1px solid rgba(15,23,42,.12);">Código del arnés</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
      <div style="margin-top:10px; opacity:.7; font-weight:800;">
        Total: <b>${escHtml(String(total))}</b>
        ${truncated ? ` · Mostrando primeros <b>${escHtml(String(items.length))}</b>` : ``}
      </div>
    `;

  }catch(e){
    body.innerHTML = `<div style="opacity:.7; font-weight:800;">Error: ${escHtml(String(e))}</div>`;
  }
}


// Exponer a window (para onclick del HTML)
window.selectSubcat = selectSubcat;
window.selectMachine = selectMachine;
window.selectPeriod = selectPeriod;
window.onPeriodValueChange = onPeriodValueChange;
window.onAppMachineChange = onAppMachineChange;
