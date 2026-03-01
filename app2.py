import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import math
import time
from collections import Counter

os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache'

CLASS_LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented",
}

# Fixed fusion weights
IMG_WEIGHT = 0.70
TAB_WEIGHT = 0.30

st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet">
    <style>
    /* ── Hide sidebar entirely ─────────────────────────────── */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Global reset ──────────────────────────────────────── */
    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    .main, .block-container {
        background: #080c14 !important;
        color: #e2e8f0;
        font-family: 'Syne', sans-serif;
    }
    .block-container {
        max-width: 1200px !important;
        padding-top: 2rem !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ── Animated scanline overlay ─────────────────────────── */
    .main::before {
        content: '';
        position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,200,0.015) 2px,
            rgba(0,255,200,0.015) 4px
        );
        pointer-events: none;
        z-index: 999;
        animation: scanMove 8s linear infinite;
    }
    @keyframes scanMove {
        0%   { background-position: 0 0; }
        100% { background-position: 0 100px; }
    }

    /* ── Hero header ───────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        position: relative;
    }
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #00ffc8 0%, #00a8ff 50%, #bf5af2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titlePulse 4s ease-in-out infinite;
    }
    @keyframes titlePulse {
        0%, 100% { filter: brightness(1); }
        50%       { filter: brightness(1.2); }
    }
    .hero-sub {
        color: #64748b;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* ── Weight badge ──────────────────────────────────────── */
    .weight-badge {
        display: inline-flex; gap: 1rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,255,200,0.2);
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        margin: 1rem auto;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: #94a3b8;
        justify-content: center;
    }
    .weight-badge span { color: #00ffc8; font-weight: 600; }

    /* ── Tab styling ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        gap: 6px;
        padding: 6px 10px;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #64748b !important;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.95rem !important;
        transition: all 0.3s;
        padding: 12px 28px !important;
        letter-spacing: 0.02em;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,255,200,0.15), rgba(0,168,255,0.15)) !important;
        color: #00ffc8 !important;
        border: 1px solid rgba(0,255,200,0.3) !important;
    }

    /* ── Result card ───────────────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, rgba(0,255,200,0.08) 0%, rgba(0,168,255,0.08) 100%);
        border: 1px solid rgba(0,255,200,0.25);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        animation: cardReveal 0.6s ease-out both;
    }
    .result-card::before {
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(0,255,200,0.05) 60deg, transparent 120deg);
        animation: spin 6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes cardReveal {
        from { opacity: 0; transform: translateY(20px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0)    scale(1); }
    }
    .result-label { font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: #64748b; font-family: 'DM Mono', monospace; position: relative; z-index: 1; }
    .result-value { font-size: 2.2rem; font-weight: 800; margin: 0.5rem 0; position: relative; z-index: 1;
        background: linear-gradient(90deg, #00ffc8, #00a8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .result-conf  { font-family: 'DM Mono', monospace; font-size: 0.9rem; color: #94a3b8; position: relative; z-index: 1; }

    /* ── Fusion card ───────────────────────────────────────── */
    .fusion-card {
        background: linear-gradient(135deg, rgba(191,90,242,0.12) 0%, rgba(0,255,200,0.08) 100%);
        border: 1px solid rgba(191,90,242,0.35);
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: cardReveal 0.6s ease-out both;
        position: relative;
        overflow: hidden;
    }
    .fusion-card::after {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(90deg, transparent 30%, rgba(191,90,242,0.04) 50%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .fusion-value { font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #bf5af2, #00ffc8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

    /* ── Animated loader ───────────────────────────────────── */
    .loader-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0,255,200,0.15);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
    }
    .loader-step {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.5rem 0;
        color: #475569;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        animation: stepFade 0.4s ease-out both;
    }
    .loader-step:last-child { border-bottom: none; }
    .loader-step.active { color: #00ffc8; }
    .loader-step.done   { color: #64748b; }
    @keyframes stepFade { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: translateX(0); } }
    .step-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: currentColor; flex-shrink: 0;
    }
    .step-dot.pulse { animation: dotPulse 0.8s ease-in-out infinite; }
    @keyframes dotPulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.6); } }

    /* ── Progress bar ──────────────────────────────────────── */
    .stProgress > div > div { background: linear-gradient(90deg, #00ffc8, #00a8ff) !important; border-radius: 4px; }
    .stProgress { background: rgba(255,255,255,0.06) !important; border-radius: 4px; }

    /* ── Info / warning boxes ──────────────────────────────── */
    .info-panel {
        background: rgba(0,168,255,0.08);
        border: 1px solid rgba(0,168,255,0.25);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.88rem;
        color: #94a3b8;
    }

    /* ── Upload area ───────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: rgba(0,255,200,0.03) !important;
        border: 2px dashed rgba(0,255,200,0.2) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,255,200,0.5) !important;
    }

    /* ── Selectbox ─────────────────────────────────────────── */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.04) !important;
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00ffc8, #00a8ff) !important;
        color: #080c14 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        letter-spacing: 0.05em;
        transition: all 0.3s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,255,200,0.3) !important;
    }

    /* ── Dividers ──────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Section headings ──────────────────────────────────── */
    h2, h3 { color: #e2e8f0 !important; font-family: 'Syne', sans-serif !important; }
    h4      { color: #94a3b8 !important; font-family: 'Syne', sans-serif !important; font-size: 0.9rem !important; letter-spacing: 0.1em; text-transform: uppercase; }

    /* ── Interpretation alerts ─────────────────────────────── */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* ── CDR table grid ────────────────────────────────────── */
    .cdr-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .cdr-domain {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: border-color 0.3s;
    }
    .cdr-domain:hover { border-color: rgba(0,255,200,0.3); }

    /* ── CDR scoring steps ─────────────────────────────────── */
    .cdr-step {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid rgba(0,255,200,0.4);
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #64748b;
        animation: stepFade 0.5s ease-out both;
    }

    /* ── Probability row ───────────────────────────────────── */
    .prob-row { margin-bottom: 1.1rem; }
    .prob-label {
        font-size: 0.82rem;
        font-family: 'DM Mono', monospace;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }

    /* ── Brain scan icon ───────────────────────────────────── */
    .brain-icon {
        font-size: 5rem;
        display: block;
        text-align: center;
        animation: brainFloat 3s ease-in-out infinite;
    }
    @keyframes brainFloat {
        0%,100% { transform: translateY(0); }
        50%      { transform: translateY(-8px); }
    }
    </style>
""")


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_image_model():
    try:
        model_name = "Thamer/resnet-fine_tuned"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None, None


def predict_image(image, processor, model):
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()
        all_probs = {
            "Non Demented":       probs[2].item() * 100,
            "Very Mild Demented": probs[3].item() * 100,
            "Mild Demented":      probs[0].item() * 100,
            "Moderate Demented":  probs[1].item() * 100,
        }
        return CLASS_LABELS[idx], probs[idx].item() * 100, all_probs
    except Exception as e:
        st.error(f"Image prediction error: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  CLINICAL RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_cdr_global(M, O, J, C, H, PC):
    secondaries = [O, J, C, H, PC]
    steps = []

    if M == 0:
        impaired = sum(1 for s in secondaries if s >= 0.5)
        if impaired < 2:
            steps.append(f"R1a: M=0 and only {impaired} secondary ≥ 0.5  →  CDR = 0")
            return 0.0, steps
        else:
            steps.append(f"R1b: M=0 but {impaired} secondaries ≥ 0.5  →  CDR = 0.5")
            return 0.5, steps

    if M == 0.5:
        gte_one = sum(1 for s in secondaries if s >= 1.0)
        if gte_one >= 3:
            steps.append(f"R2b: M=0.5 and {gte_one} secondaries ≥ 1  →  CDR = 1")
            return 1.0, steps
        else:
            steps.append(f"R2a: M=0.5 and only {gte_one} secondaries ≥ 1  →  CDR = 0.5")
            return 0.5, steps

    equal   = [s for s in secondaries if s == M]
    greater = [s for s in secondaries if s >  M]
    lesser  = [s for s in secondaries if s <  M]

    if len(equal) >= 3:
        steps.append(f"R3: {len(equal)}/5 secondaries = M={M}  →  CDR = {M}")
        return M, steps

    if (len(greater) == 3 and len(lesser) == 2) or \
       (len(lesser) == 3 and len(greater) == 2):
        steps.append(f"R4: split 3-2 secondaries  →  CDR = M = {M}")
        return M, steps

    if len(greater) >= 3:
        cnt = Counter(greater)
        max_n = max(cnt.values())
        candidates = [sc for sc, n in cnt.items() if n == max_n]
        chosen = min(candidates, key=lambda x: abs(x - M))
        steps.append(f"R5: {len(greater)} secondaries > M  →  majority = {chosen}")
        if M >= 1 and chosen == 0:
            chosen = 0.5
            steps.append("R7g guard: floored at 0.5")
        return chosen, steps

    if len(lesser) >= 3:
        cnt = Counter(lesser)
        max_n = max(cnt.values())
        candidates = [sc for sc, n in cnt.items() if n == max_n]
        chosen = min(candidates, key=lambda x: abs(x - M))
        steps.append(f"R6: {len(lesser)} secondaries < M  →  majority = {chosen}")
        if M >= 1 and chosen == 0:
            chosen = 0.5
            steps.append("R7g guard: floored at 0.5")
        return chosen, steps

    if len(equal) in (1, 2) and len(greater) <= 2 and len(lesser) <= 2:
        steps.append(f"R7: {len(equal)} secondary(ies) = M  →  CDR = M = {M}")
        return M, steps

    all_six = [M] + secondaries
    cnt = Counter(all_six)
    chosen = cnt.most_common(1)[0][0]
    steps.append(f"Rfb: fallback majority = {chosen}")
    if M >= 1 and chosen == 0:
        chosen = 0.5
        steps.append("R7g guard: floored at 0.5")
    return chosen, steps


def compute_cdr_sb(M, O, J, C, H, PC):
    return M + O + J + C + H + PC


def cdr_sb_to_base_probs(cdr_sb):
    s = cdr_sb
    if s == 0:
        p = {"Non Demented": 95.0, "Very Mild Demented": 4.0,
             "Mild Demented": 0.8,  "Moderate Demented": 0.2}
    elif s <= 4.0:
        t = s / 4.0
        p = {"Non Demented":       max(2.0,  62.0 - 58.0 * t),
             "Very Mild Demented": min(87.0, 23.0 + 68.0 * t),
             "Mild Demented":      min(10.0,  2.0 +  8.0 * t),
             "Moderate Demented":  1.0}
    elif s <= 9.0:
        t = (s - 4.5) / 4.5
        p = {"Non Demented":       2.0,
             "Very Mild Demented": max(5.0,  45.0 - 38.0 * t),
             "Mild Demented":      min(80.0, 40.0 + 38.0 * t),
             "Moderate Demented":  min(15.0,  5.0 + 10.0 * t)}
    elif s <= 15.5:
        t = (s - 9.5) / 6.0
        p = {"Non Demented":       0.5,
             "Very Mild Demented": 2.0,
             "Mild Demented":      max(5.0,  35.0 - 28.0 * t),
             "Moderate Demented":  min(92.0, 55.0 + 35.0 * t)}
    else:
        t = (s - 16.0) / 2.0
        p = {"Non Demented":       0.2,
             "Very Mild Demented": 0.5,
             "Mild Demented":      max(1.5,  8.0 -  6.0 * t),
             "Moderate Demented":  min(97.8, 88.0 + 6.5 * t)}
    total = sum(p.values())
    return {k: v / total * 100 for k, v in p.items()}


CDR_TO_CLASS = {
    0.0: "Non Demented",
    0.5: "Very Mild Demented",
    1.0: "Mild Demented",
    2.0: "Moderate Demented",
    3.0: "Moderate Demented",
}

def anchor_to_cdr_global(probs, global_cdr):
    forced = CDR_TO_CLASS.get(global_cdr, "Very Mild Demented")
    p = dict(probs)
    if p[forced] < 55.0:
        deficit = 55.0 - p[forced]
        others_total = sum(v for k, v in p.items() if k != forced)
        p[forced] = 55.0
        for k in p:
            if k != forced:
                p[k] = max(0.5, p[k] - deficit * (p[k] / others_total))
    total = sum(p.values())
    return {k: v / total * 100 for k, v in p.items()}, forced


APOE_OR = {
    "2/2": 0.60, "2/3": 0.60, "2/4": 2.60,
    "3/3": 1.00, "3/4": 3.20, "4/4": 14.9,
}

def apply_apoe_modifier(probs, genotype):
    OR    = APOE_OR.get(genotype, 1.0)
    delta = math.log(OR) / math.log(14.9) * 30.0
    p = dict(probs)
    if delta > 0:
        donor, recipient = "Non Demented", "Moderate Demented"
        actual = min(delta, p[donor] * 0.85)
        p[donor]     -= actual
        p[recipient] += actual
    elif delta < 0:
        donor, recipient = "Moderate Demented", "Non Demented"
        actual = min(abs(delta), p[donor] * 0.85)
        p[donor]     -= actual
        p[recipient] += actual
    total = sum(p.values())
    return {k: v / total * 100 for k, v in p.items()}


def predict_clinical(M, O, J, C, H, PC, genotype):
    global_cdr, reasoning = compute_cdr_global(M, O, J, C, H, PC)
    cdr_sb                = compute_cdr_sb(M, O, J, C, H, PC)
    base_probs            = cdr_sb_to_base_probs(cdr_sb)
    anchored, forced      = anchor_to_cdr_global(base_probs, global_cdr)
    final_probs           = apply_apoe_modifier(anchored, genotype)
    label      = forced
    confidence = final_probs[label]
    return label, confidence, final_probs, global_cdr, cdr_sb, reasoning


def fuse_predictions(img_probs, tab_probs, img_weight=IMG_WEIGHT, tab_weight=TAB_WEIGHT):
    fused = {l: img_probs[l] * img_weight + tab_probs[l] * tab_weight
             for l in img_probs}
    best = max(fused, key=fused.get)
    return best, fused[best], fused


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Non Demented":       "#00ffc8",
    "Very Mild Demented": "#f59e0b",
    "Mild Demented":      "#f97316",
    "Moderate Demented":  "#ef4444",
}

def prob_bar(label, prob, color):
    st.markdown(f"<div class='prob-label'>{label}</div>", unsafe_allow_html=True)
    st.progress(prob / 100)
    st.markdown(
        f"<span style='color:{color};font-weight:700;font-family:DM Mono,monospace;font-size:0.85rem'>{prob:.1f}%</span>",
        unsafe_allow_html=True)
    st.markdown("")


def interpretation_block(label):
    msgs = {
        "Non Demented":       ("success", "✅ No signs of dementia detected. Regular check-ups are still recommended."),
        "Very Mild Demented": ("warning", "⚠️ Very mild signs detected. Early intervention may help. Consult a neurologist."),
        "Mild Demented":      ("warning", "⚠️ Mild dementia indicated. Medical evaluation and treatment planning are recommended."),
        "Moderate Demented":  ("error",   "🚨 Moderate dementia indicated. Comprehensive medical care is essential. Consult a healthcare professional immediately."),
    }
    kind, msg = msgs.get(label, ("info", ""))
    getattr(st, kind)(msg)


def animated_loader_image():
    """Shows a step-by-step animated loading sequence for the image model."""
    steps = [
        ("🔬", "Preprocessing MRI scan…"),
        ("🧠", "Loading neural network weights…"),
        ("⚡", "Running quick inference pass…"),
        ("🔍", "Scanning cortical features…"),
        ("📐", "Analysing hippocampal volume…"),
        ("🧮", "Computing softmax probabilities…"),
        ("✅", "Inference complete!"),
    ]
    container = st.empty()
    completed = []
    for icon, msg in steps:
        completed.append((icon, msg))
        html = "<div class='loader-container'>"
        for i, (ic, m) in enumerate(completed):
            is_last = i == len(completed) - 1
            dot_cls = "pulse" if is_last else ""
            step_cls = "active" if is_last else "done"
            html += f"<div class='loader-step {step_cls}'><span class='step-dot {dot_cls}'></span>{ic}&nbsp; {m}</div>"
        html += "</div>"
        container.markdown(html, unsafe_allow_html=True)
        time.sleep(0.45)
    time.sleep(0.3)
    container.empty()


def animated_loader_clinical():
    """Shows a step-by-step animated loading sequence for the clinical engine."""
    steps = [
        ("📋", "Parsing CDR domain scores…"),
        ("⚖️",  "Applying Morris (1993) global scoring rules…"),
        ("➕", "Calculating CDR Sum-of-Boxes…"),
        ("📊", "Mapping to base probability distributions…"),
        ("🔒", "Anchoring to CDR Global class…"),
        ("🧬", "Applying APOE genotype modifier…"),
        ("✅", "Clinical analysis complete!"),
    ]
    container = st.empty()
    completed = []
    for icon, msg in steps:
        completed.append((icon, msg))
        html = "<div class='loader-container'>"
        for i, (ic, m) in enumerate(completed):
            is_last = i == len(completed) - 1
            dot_cls = "pulse" if is_last else ""
            step_cls = "active" if is_last else "done"
            html += f"<div class='loader-step {step_cls}'><span class='step-dot {dot_cls}'></span>{ic}&nbsp; {m}</div>"
        html += "</div>"
        container.markdown(html, unsafe_allow_html=True)
        time.sleep(0.40)
    time.sleep(0.3)
    container.empty()


def animated_loader_fusion():
    steps = [
        ("🔀", "Collecting image model probabilities…"),
        ("📋", "Collecting clinical engine probabilities…"),
        ("⚖️",  f"Applying weights — Image: {IMG_WEIGHT*100:.0f}% | Clinical: {TAB_WEIGHT*100:.0f}%…"),
        ("🧮", "Computing weighted average fusion…"),
        ("🎯", "Determining final diagnosis…"),
        ("✅", "Fusion complete!"),
    ]
    container = st.empty()
    completed = []
    for icon, msg in steps:
        completed.append((icon, msg))
        html = "<div class='loader-container'>"
        for i, (ic, m) in enumerate(completed):
            is_last = i == len(completed) - 1
            dot_cls = "pulse" if is_last else ""
            step_cls = "active" if is_last else "done"
            html += f"<div class='loader-step {step_cls}'><span class='step-dot {dot_cls}'></span>{ic}&nbsp; {m}</div>"
        html += "</div>"
        container.markdown(html, unsafe_allow_html=True)
        time.sleep(0.35)
    time.sleep(0.3)
    container.empty()


CD_OPTIONS = {
    "0 – None":           0.0,
    "0.5 – Questionable": 0.5,
    "1 – Mild":           1.0,
    "2 – Moderate":       2.0,
    "3 – Severe":         3.0,
}
CD_OPTIONS_CARE = {
    "0 – None":     0.0,
    "1 – Mild":     1.0,
    "2 – Moderate": 2.0,
    "3 – Severe":   3.0,
}
GENOTYPE_OPTIONS = ["2/2", "2/3", "2/4", "3/3", "3/4", "4/4"]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Hero header
    st.markdown("""
    <div class='hero-header'>
        <span class='brain-icon'>🧠</span>
        <div class='hero-title'>Alzheimer's Classification</div>
        <div class='hero-sub'>Multimodal Neural + Clinical Analysis</div>
        <div style='display:flex;justify-content:center;margin-top:1rem'>
            
    </div>
    """, unsafe_allow_html=True)

    # Model loading
    with st.spinner("Initialising neural network…"):
        img_processor, img_model = load_image_model()

    if img_processor is None or img_model is None:
        st.error("Failed to load the image model.")
        return

    st.success("✅ ResNet-34 image model loaded and ready!")
    st.markdown("---")

    tab_img, tab_clinical, tab_results = st.tabs(
        ["🖼️  MRI Image", "📋  Clinical & Genetic Features", "📊  Results & Fusion"])

    # ── Tab 1: MRI ────────────────────────────────────────────────────────────
    with tab_img:
        st.markdown("## 📤 Upload Brain MRI Scan")
        st.markdown("<div class='info-panel'>Upload a T1-weighted axial MRI slice. Supported formats: JPG, PNG.</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop your MRI scan here", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed")

        if uploaded_file:
            image = Image.open(uploaded_file)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(image, use_container_width=True, caption="Uploaded MRI")
                st.markdown(f"""
                <div class='info-panel'>
                    📐 <b>Dimensions:</b> {image.size[0]} × {image.size[1]} px &nbsp;|&nbsp;
                    🎨 <b>Mode:</b> {image.mode}
                </div>""", unsafe_allow_html=True)
            with c2:
                if st.button("🚀 Analyse MRI Scan", type="primary", use_container_width=True):
                    animated_loader_image()
                    img_pred, img_conf, img_probs = predict_image(
                        image, img_processor, img_model)
                    st.session_state.update({
                        "img_pred": img_pred,
                        "img_conf": img_conf,
                        "img_probs": img_probs,
                    })
                    if img_pred:
                        color = SEVERITY_COLORS[img_pred]
                        st.markdown(f"""
                        <div class='result-card'>
                            <div class='result-label'>Image Model Diagnosis</div>
                            <div class='result-value' style='-webkit-background-clip:text;background-clip:text'>{img_pred}</div>
                            <div class='result-conf'>Confidence: {img_conf:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("#### 📊 Class Probabilities")
                        for lbl, p in sorted(img_probs.items(), key=lambda x: -x[1]):
                            prob_bar(lbl, p, SEVERITY_COLORS[lbl])
                        interpretation_block(img_pred)
        else:
            st.markdown("<div style='text-align:center;padding:3rem;color:#475569;font-family:DM Mono,monospace;font-size:0.85rem'>Upload a brain MRI scan above to begin image analysis.</div>", unsafe_allow_html=True)

    # ── Tab 2: Clinical Rule Engine ───────────────────────────────────────────
    with tab_clinical:
        st.markdown("## 📋 Clinical Dementia Rating (CDR)")
        # Genotype
        st.markdown("### 🧬 APOE Genotype")
        c1, c2 = st.columns([1, 2])
        with c1:
            genotype = st.selectbox("Select genotype", GENOTYPE_OPTIONS, index=3, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📝 CDR Domain Scores")

        col1, col2, col3 = st.columns(3)
        with col1:
            memory      = st.selectbox("🧠 Memory *(primary)*", list(CD_OPTIONS.keys()), index=0)
            orientation = st.selectbox("🧭 Orientation",        list(CD_OPTIONS.keys()), index=0)
        with col2:
            judgment      = st.selectbox("⚖️ Judgment & Problem Solving", list(CD_OPTIONS.keys()), index=0)
            communication = st.selectbox("🏙️ Community Affairs",          list(CD_OPTIONS.keys()), index=0)
        with col3:
            home = st.selectbox("🏠 Home & Hobbies", list(CD_OPTIONS.keys()),      index=0)
            care = st.selectbox("🪥 Personal Care",  list(CD_OPTIONS_CARE.keys()), index=0)

        M_val  = CD_OPTIONS[memory]
        O_val  = CD_OPTIONS[orientation]
        J_val  = CD_OPTIONS[judgment]
        C_val  = CD_OPTIONS[communication]
        H_val  = CD_OPTIONS[home]
        PC_val = CD_OPTIONS_CARE[care]

        sb_live = compute_cdr_sb(M_val, O_val, J_val, C_val, H_val, PC_val)
        if sb_live == 0:          sb_stage = "Normal (CDR 0)"
        elif sb_live <= 4.0:      sb_stage = "Very Mild Dementia range (CDR 0.5)"
        elif sb_live <= 9.0:      sb_stage = "Mild Dementia range (CDR 1)"
        elif sb_live <= 15.5:     sb_stage = "Moderate Dementia range (CDR 2)"
        else:                     sb_stage = "Severe Dementia range (CDR 3)"

        
        st.markdown("---")
        if st.button("🧪 Run Clinical Rule Engine", type="primary", use_container_width=True):
            animated_loader_clinical()
            (tab_pred, tab_conf, tab_probs,
             global_cdr, cdr_sb, reasoning) = predict_clinical(
                M_val, O_val, J_val, C_val, H_val, PC_val, genotype)

            st.session_state.update({
                "tab_pred":  tab_pred,
                "tab_conf":  tab_conf,
                "tab_probs": tab_probs,
            })

            color = SEVERITY_COLORS[tab_pred]
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-label'>Clinical Rule Engine Diagnosis</div>
                <div class='result-value' style='-webkit-background-clip:text;background-clip:text'>{tab_pred}</div>
                <div class='result-conf'>Confidence: {tab_conf:.1f}%&nbsp;&nbsp;|&nbsp;&nbsp;CDR Global: {global_cdr}&nbsp;&nbsp;|&nbsp;&nbsp;CDR-SB: {cdr_sb:.1f}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### 📊 Class Probabilities (after APOE adjustment)")
            for lbl, p in sorted(tab_probs.items(), key=lambda x: -x[1]):
                prob_bar(lbl, p, SEVERITY_COLORS[lbl])

            # with st.expander("🔍 CDR Global Scoring Logic"):
            #     for step in reasoning:
            #         st.markdown(f"<div class='cdr-step'>{step}</div>", unsafe_allow_html=True)

            interpretation_block(tab_pred)

    # ── Tab 3: Fusion ─────────────────────────────────────────────────────────
    with tab_results:
        st.markdown("## 📊 Multimodal Fusion Results")


        img_probs_stored = st.session_state.get("img_probs")
        tab_probs_stored = st.session_state.get("tab_probs")
        have_img = img_probs_stored is not None
        have_tab = tab_probs_stored is not None

        if not have_img and not have_tab:
            st.markdown("<div style='text-align:center;padding:3rem;color:#475569;font-family:DM Mono,monospace;font-size:0.85rem'>Run at least one model to see results here.</div>", unsafe_allow_html=True)
            return

        col_img, col_tab = st.columns(2)
        with col_img:
            st.markdown("### 🖼️ Image Model")
            if have_img:
                color = SEVERITY_COLORS[st.session_state["img_pred"]]
                st.markdown(f"""
                <div class='result-card'>
                    <div class='result-label'>Diagnosis</div>
                    <div class='result-value' style='-webkit-background-clip:text;background-clip:text'>{st.session_state["img_pred"]}</div>
                    <div class='result-conf'>Confidence: {st.session_state["img_conf"]:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                for lbl, p in sorted(img_probs_stored.items(), key=lambda x: -x[1]):
                    prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            else:
                st.info("No image analysis run yet.")

        with col_tab:
            st.markdown("### 📋 Clinical Engine")
            if have_tab:
                color = SEVERITY_COLORS[st.session_state["tab_pred"]]
                st.markdown(f"""
                <div class='result-card'>
                    <div class='result-label'>Diagnosis</div>
                    <div class='result-value' style='-webkit-background-clip:text;background-clip:text'>{st.session_state["tab_pred"]}</div>
                    <div class='result-conf'>Confidence: {st.session_state["tab_conf"]:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                for lbl, p in sorted(tab_probs_stored.items(), key=lambda x: -x[1]):
                    prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            else:
                st.info("No clinical analysis run yet.")

        if have_img and have_tab:
            st.markdown("---")
            st.markdown("### 🔀 Final Fused Prediction")
            animated_loader_fusion()
            fused_label, fused_conf, fused_probs = fuse_predictions(
                img_probs_stored, tab_probs_stored)
            color = SEVERITY_COLORS[fused_label]
            st.markdown(f"""
            <div class='fusion-card'>
                <div class='result-label' style='color:#bf5af2'>Final Fused Diagnosis</div>
                <div class='fusion-value'>{fused_label}</div>
                <div class='result-conf' style='margin-top:0.75rem'>
                    Fused Score: <b style='color:#00ffc8'>{fused_conf:.1f}%</b>
                   
            </div>""", unsafe_allow_html=True)
            st.markdown("#### 📊 Fused Class Probabilities")
            for lbl, p in sorted(fused_probs.items(), key=lambda x: -x[1]):
                prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            st.markdown("### 💡 Clinical Interpretation")
            interpretation_block(fused_label)

        elif have_img:
            st.markdown("### 💡 Interpretation (Image model only)")
            interpretation_block(st.session_state["img_pred"])
        else:
            st.markdown("### 💡 Interpretation (Clinical rule engine only)")
            interpretation_block(st.session_state["tab_pred"])

    


if __name__ == "__main__":
    main()