import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import math
from collections import Counter

os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache'

CLASS_LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented",
}

st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white;
        text-align: center; margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-label { font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; }
    .prediction-value { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .confidence-text  { font-size: 1.2rem; opacity: 0.9; }
    .info-box {
        background-color: #e8f4f8; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #3498db; margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #ffc107; margin: 1rem 0;
    }
    .fusion-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem; border-radius: 15px; color: white;
        text-align: center; margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .cdr-step-active {
        background-color: #d4edda; padding: 0.6rem 1rem;
        border-radius: 8px; border-left: 4px solid #28a745;
        margin: 0.4rem 0; font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


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

# ── Step 1: Official CDR Global Score ─────────────────────────────────────
# Source: Morris JC (1993) Neurology 43(11):2412-2414
#         Washington University Knight ADRC Scoring Rules
#         https://knightadrc.wustl.edu/professionals-clinicians/
#                 cdr-dementia-staging-instrument/cdr-scoring-rules/
def compute_cdr_global(M, O, J, C, H, PC):
    """
    Returns (global_cdr_score, list_of_reasoning_steps).

    Official rules applied in strict order:
      R1a  M=0, <2 secondaries ≥0.5         → CDR = 0
      R1b  M=0, ≥2 secondaries ≥0.5         → CDR = 0.5
      R2a  M=0.5, <3 secondaries ≥1          → CDR = 0.5
      R2b  M=0.5, ≥3 secondaries ≥1          → CDR = 1
      R3   ≥3 secondaries == M               → CDR = M
      R4   3 on one side AND 2 on other      → CDR = M
      R5   ≥3 secondaries > M                → CDR = majority (closest-to-M tie-break)
      R6   ≥3 secondaries < M                → CDR = majority (closest-to-M tie-break)
      R7   1–2 equal M, ≤2 on each side      → CDR = M
      R7g  M≥1 guard: CDR cannot be 0        → floor at 0.5
      Rfb  Fallback: majority of all 6
    """
    secondaries = [O, J, C, H, PC]
    steps = []

    # R1 ── M = 0
    if M == 0:
        impaired = sum(1 for s in secondaries if s >= 0.5)
        if impaired < 2:
            steps.append(f"R1a: M=0 and only {impaired} secondary ≥ 0.5  →  CDR = 0")
            return 0.0, steps
        else:
            steps.append(f"R1b: M=0 but {impaired} secondaries ≥ 0.5  →  CDR = 0.5")
            return 0.5, steps

    # R2 ── M = 0.5
    if M == 0.5:
        gte_one = sum(1 for s in secondaries if s >= 1.0)
        if gte_one >= 3:
            steps.append(f"R2b: M=0.5 and {gte_one} secondaries ≥ 1  →  CDR = 1")
            return 1.0, steps
        else:
            steps.append(f"R2a: M=0.5 and only {gte_one} secondaries ≥ 1  →  CDR = 0.5 (cannot be 0)")
            return 0.5, steps

    # Classify secondaries
    equal   = [s for s in secondaries if s == M]
    greater = [s for s in secondaries if s >  M]
    lesser  = [s for s in secondaries if s <  M]

    # R3 ── ≥3 secondaries equal M
    if len(equal) >= 3:
        steps.append(f"R3: {len(equal)}/5 secondaries = M={M}  →  CDR = M = {M}")
        return M, steps

    # R4 ── 3 on one side, 2 on other → CDR = M
    if (len(greater) == 3 and len(lesser) == 2) or \
       (len(lesser) == 3 and len(greater) == 2):
        steps.append(
            f"R4: 3 secondaries on one side and 2 on the other  →  CDR = M = {M}")
        return M, steps

    # R5 ── ≥3 secondaries > M
    if len(greater) >= 3:
        cnt = Counter(greater)
        max_n = max(cnt.values())
        candidates = [sc for sc, n in cnt.items() if n == max_n]
        chosen = min(candidates, key=lambda x: abs(x - M))
        steps.append(
            f"R5: {len(greater)} secondaries > M  →  majority among them = {chosen}  →  CDR = {chosen}")
        if M >= 1 and chosen == 0:
            chosen = 0.5
            steps.append("R7g (guard): M≥1 → CDR floored at 0.5")
        return chosen, steps

    # R6 ── ≥3 secondaries < M
    if len(lesser) >= 3:
        cnt = Counter(lesser)
        max_n = max(cnt.values())
        candidates = [sc for sc, n in cnt.items() if n == max_n]
        chosen = min(candidates, key=lambda x: abs(x - M))
        steps.append(
            f"R6: {len(lesser)} secondaries < M  →  majority among them = {chosen}  →  CDR = {chosen}")
        if M >= 1 and chosen == 0:
            chosen = 0.5
            steps.append("R7g (guard): M≥1 → CDR floored at 0.5")
        return chosen, steps

    # R7 ── 1–2 equal M, ≤2 on either side
    if len(equal) in (1, 2) and len(greater) <= 2 and len(lesser) <= 2:
        steps.append(
            f"R7: {len(equal)} secondary(ies) = M, ≤2 on each side  →  CDR = M = {M}")
        return M, steps

    # Rfb ── Fallback: majority of all 6
    all_six = [M] + secondaries
    cnt = Counter(all_six)
    chosen = cnt.most_common(1)[0][0]
    steps.append(f"Rfb: fallback majority of all 6 = {chosen}  →  CDR = {chosen}")
    if M >= 1 and chosen == 0:
        chosen = 0.5
        steps.append("R7g (guard): M≥1 → CDR floored at 0.5")
    return chosen, steps


# ── Step 2: CDR Sum-of-Boxes (CDR-SB) ─────────────────────────────────────
# The CDR-SB (0–18) is the arithmetic sum of all 6 domain scores.
# Validated staging cut-points (O'Bryant et al. 2008 / Cosentino et al. 2011):
#   SB = 0           → CDR 0   Normal
#   SB 0.5–4.0       → CDR 0.5 Very Mild
#   SB 4.5–9.0       → CDR 1   Mild
#   SB 9.5–15.5      → CDR 2   Moderate
#   SB 16.0–18.0     → CDR 3   Severe
def compute_cdr_sb(M, O, J, C, H, PC):
    return M + O + J + C + H + PC


def cdr_sb_to_base_probs(cdr_sb):
    """
    Produces smooth probability distributions over the four classes
    using piece-wise linear interpolation within each CDR-SB staging band.
    This gives graded confidence rather than a hard step function.
    """
    s = cdr_sb

    if s == 0:
        p = {"Non Demented": 95.0, "Very Mild Demented": 4.0,
             "Mild Demented": 0.8,  "Moderate Demented": 0.2}

    elif s <= 4.0:           # CDR 0.5 band (0.5–4.0)
        t = s / 4.0          # 0 → 1 across the band
        p = {"Non Demented":       max(2.0,  62.0 - 58.0 * t),
             "Very Mild Demented": min(87.0, 23.0 + 68.0 * t),
             "Mild Demented":      min(10.0,  2.0 +  8.0 * t),
             "Moderate Demented":  1.0}

    elif s <= 9.0:           # CDR 1 band (4.5–9.0)
        t = (s - 4.5) / 4.5
        p = {"Non Demented":       2.0,
             "Very Mild Demented": max(5.0,  45.0 - 38.0 * t),
             "Mild Demented":      min(80.0, 40.0 + 38.0 * t),
             "Moderate Demented":  min(15.0,  5.0 + 10.0 * t)}

    elif s <= 15.5:          # CDR 2 band (9.5–15.5)
        t = (s - 9.5) / 6.0
        p = {"Non Demented":       0.5,
             "Very Mild Demented": 2.0,
             "Mild Demented":      max(5.0,  35.0 - 28.0 * t),
             "Moderate Demented":  min(92.0, 55.0 + 35.0 * t)}

    else:                    # CDR 3 band (16–18)
        t = (s - 16.0) / 2.0
        p = {"Non Demented":       0.2,
             "Very Mild Demented": 0.5,
             "Mild Demented":      max(1.5,  8.0 -  6.0 * t),
             "Moderate Demented":  min(97.8, 88.0 + 6.5 * t)}

    total = sum(p.values())
    return {k: v / total * 100 for k, v in p.items()}


# ── Step 3: Anchor probabilities to the CDR Global class ─────────────────
CDR_TO_CLASS = {
    0.0: "Non Demented",
    0.5: "Very Mild Demented",
    1.0: "Mild Demented",
    2.0: "Moderate Demented",
    3.0: "Moderate Demented",
}

def anchor_to_cdr_global(probs, global_cdr):
    """
    If the CDR Global class is not already dominant in the SB-derived
    probability vector, boost it to ≥55 % (mass taken proportionally
    from the other classes).  This ensures the official score always wins.
    """
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


# ── Step 4: APOE genotype modifier ────────────────────────────────────────
# Published odds ratios vs ε3/ε3 baseline (Farrer et al. JAMA 1997 meta-analysis;
# corroborated by ScienceDirect APOE review 2021 & Mayo Clinic 2024):
#   ε2/ε2  OR ≈ 0.60  (≈ −40 % AD risk vs ε3/ε3)  — protective
#   ε2/ε3  OR ≈ 0.60                               — protective
#   ε2/ε4  OR ≈ 2.60  (one protective + one risk allele)
#   ε3/ε3  OR = 1.00  — neutral baseline
#   ε3/ε4  OR ≈ 3.20  (2–3× increased risk)
#   ε4/ε4  OR ≈ 14.9  (10–15× increased risk)
APOE_OR = {
    "2/2": 0.60,
    "2/3": 0.60,
    "2/4": 2.60,
    "3/3": 1.00,
    "3/4": 3.20,
    "4/4": 14.9,
}

def apply_apoe_modifier(probs, genotype):
    """
    Shifts probability mass toward 'Moderate Demented' for risk genotypes
    and toward 'Non Demented' for protective ones, using a log-OR scale
    normalised so ε4/ε4 (OR=14.9) → +30 pp shift and
    ε2/ε2 (OR=0.6) → −5.7 pp shift.
    Mass is conserved: donor is the opposite end of the spectrum.
    """
    OR    = APOE_OR.get(genotype, 1.0)
    delta = math.log(OR) / math.log(14.9) * 30.0   # pp shift

    p = dict(probs)

    if delta > 0:    # risk genotype: Non → Moderate
        donor, recipient = "Non Demented", "Moderate Demented"
        actual = min(delta, p[donor] * 0.85)
        p[donor]     -= actual
        p[recipient] += actual
    elif delta < 0:  # protective genotype: Moderate → Non
        donor, recipient = "Moderate Demented", "Non Demented"
        actual = min(abs(delta), p[donor] * 0.85)
        p[donor]     -= actual
        p[recipient] += actual

    total = sum(p.values())
    return {k: v / total * 100 for k, v in p.items()}


# ── Full pipeline ─────────────────────────────────────────────────────────
def predict_clinical(M, O, J, C, H, PC, genotype):
    global_cdr, reasoning = compute_cdr_global(M, O, J, C, H, PC)
    cdr_sb                = compute_cdr_sb(M, O, J, C, H, PC)
    base_probs            = cdr_sb_to_base_probs(cdr_sb)
    anchored, forced      = anchor_to_cdr_global(base_probs, global_cdr)
    final_probs           = apply_apoe_modifier(anchored, genotype)

    label      = forced
    confidence = final_probs[label]
    return label, confidence, final_probs, global_cdr, cdr_sb, reasoning


# ─────────────────────────────────────────────────────────────────────────────
#  FUSION
# ─────────────────────────────────────────────────────────────────────────────
def fuse_predictions(img_probs, tab_probs, img_weight=0.5, tab_weight=0.5):
    fused = {l: img_probs[l] * img_weight + tab_probs[l] * tab_weight
             for l in img_probs}
    best = max(fused, key=fused.get)
    return best, fused[best], fused


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Non Demented":       "#27ae60",
    "Very Mild Demented": "#f39c12",
    "Mild Demented":      "#e67e22",
    "Moderate Demented":  "#e74c3c",
}

def prob_bar(label, prob, color):
    st.markdown(f"**{label}**")
    st.progress(prob / 100)
    st.markdown(
        f"<span style='color:{color};font-weight:bold'>{prob:.1f}%</span>",
        unsafe_allow_html=True)
    st.markdown("")


def interpretation_block(label):
    msgs = {
        "Non Demented":       ("success", "No signs of dementia detected. Regular check-ups are still recommended."),
        "Very Mild Demented": ("warning", "Very mild signs of dementia. Early intervention may help. Consult a neurologist."),
        "Mild Demented":      ("warning", "Mild dementia indicated. Medical evaluation and treatment planning are recommended."),
        "Moderate Demented":  ("error",   "Moderate dementia indicated. Comprehensive medical care is essential. Please consult healthcare professionals immediately."),
    }
    kind, msg = msgs.get(label, ("info", ""))
    getattr(st, kind)(msg)


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
APOE_DESCRIPTIONS = {
    "2/2": "ε2/ε2 — Protective  (OR ≈ 0.6 vs ε3/ε3, −40 % AD risk)",
    "2/3": "ε2/ε3 — Protective  (OR ≈ 0.6 vs ε3/ε3, −40 % AD risk)",
    "2/4": "ε2/ε4 — Slightly elevated  (OR ≈ 2.6, one risk + one protective allele)",
    "3/3": "ε3/ε3 — Neutral baseline  (OR = 1.0, ~10–15 % lifetime risk)",
    "3/4": "ε3/ε4 — 2–3× increased risk  (OR ≈ 3.2, ~20–25 % lifetime risk by 85)",
    "4/4": "ε4/ε4 — 10–15× increased risk  (OR ≈ 14.9, ~30–55 % lifetime risk by 85)",
}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 Alzheimer's Disease Classification")
    st.markdown(
        "<p style='text-align:center;color:#7f8c8d;font-size:1.2rem'>"
        "Multimodal Analysis: MRI Imaging + Clinical & Genetic Features</p>",
        unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **🖼 Image Model**
        - Base: microsoft/resnet-34
        - Dataset: Falah/Alzheimer_MRI
        - Accuracy: 92.19 %

        **📋 Clinical Rule Engine (no ML)**
        - CDR Global: Morris (1993) / WashU Knight ADRC
        - CDR-SB staging: O'Bryant et al. (2008)
        - APOE OR modifier: Farrer et al. (1997)

        **Fusion:** Weighted average of class probabilities.

        | CDR | Class |
        |-----|-------|
        | 0   | 🟢 Non Demented |
        | 0.5 | 🟡 Very Mild Demented |
        | 1   | 🟠 Mild Demented |
        | 2–3 | 🔴 Moderate Demented |
        """)
        st.markdown("---")
        st.markdown("**⚠️ Research & educational use only.**")
        st.markdown("---")
        st.subheader("⚖️ Fusion Weights")
        img_w = st.slider("Image model weight", 0.0, 1.0, 0.5, 0.05)
        tab_w = round(1.0 - img_w, 2)
        st.markdown(f"Clinical engine weight: **{tab_w}**")

    
    with st.spinner("Loading image model…"):
        img_processor, img_model = load_image_model()

    if img_processor is None or img_model is None:
        st.error("Failed to load the image model.")
        return

    st.success("✅ Image model loaded!")
    st.markdown("---")

    tab_img, tab_clinical, tab_results = st.tabs(
        ["🖼️ MRI Image", "📋 Clinical & Genetic Features", "📊 Results & Fusion"])

    # ── Tab 1: MRI ────────────────────────────────────────────────────────────
    with tab_img:
        st.header("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image (JPG / PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, use_container_width=True, caption="Uploaded MRI")
                st.markdown(f"""
                <div class='info-box' style="color: red">
                    <strong>Image Details:</strong><br>
                    Size: {image.size[0]} × {image.size[1]} px &nbsp;|&nbsp;
                    Mode: {image.mode}
                </div>""", unsafe_allow_html=True)
            with c2:
                if st.button("🚀 Analyse MRI", type="primary", use_container_width=True):
                    with st.spinner("Analysing…"):
                        img_pred, img_conf, img_probs = predict_image(
                            image, img_processor, img_model)
                        st.session_state.update({
                            "img_pred": img_pred,
                            "img_conf": img_conf,
                            "img_probs": img_probs,
                        })
                    if img_pred:
                        st.markdown(f"""
                        <div class='result-box'>
                            <div class='prediction-label'>Image Model Diagnosis:</div>
                            <div class='prediction-value'>{img_pred}</div>
                            <div class='confidence-text'>Confidence: {img_conf:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("#### Class Probabilities")
                        for lbl, p in sorted(img_probs.items(), key=lambda x: -x[1]):
                            prob_bar(lbl, p, SEVERITY_COLORS[lbl])
        else:
            st.info("👆 Upload a brain MRI scan to begin image analysis.")

    # ── Tab 2: Clinical Rule Engine ───────────────────────────────────────────
    with tab_clinical:
        st.header("📋 Clinical Dementia Rating (CDR) & APOE Genotype")
        st.markdown("""
        Fill in the six CDR domain scores and the APOE genotype, then click
        **Run Clinical Rule Engine**.

        """)

        # Genotype row
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("🧬 APOE Genotype")
            genotype = st.selectbox("Select genotype", GENOTYPE_OPTIONS, index=3)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            OR_val = APOE_OR.get(genotype, 1.0)
            log_d  = math.log(OR_val) / math.log(14.9) * 30.0
            direction = "⬆ shift toward severity" if log_d > 0 else ("⬇ shift toward protection" if log_d < 0 else "no shift")
            
        st.markdown("---")
        st.subheader("📝 CDR Sub-scores")
        st.caption("Rate each domain independently based on cognitive decline (not physical limitation).")

        col1, col2, col3 = st.columns(3)
        with col1:
            memory        = st.selectbox("🧠 Memory *(primary domain)*",
                                         list(CD_OPTIONS.keys()), index=0)
            orientation   = st.selectbox("🧭 Orientation",
                                         list(CD_OPTIONS.keys()), index=0)
        with col2:
            judgment      = st.selectbox("⚖️ Judgment & Problem Solving",
                                         list(CD_OPTIONS.keys()), index=0)
            communication = st.selectbox("🏙️ Community Affairs",
                                         list(CD_OPTIONS.keys()), index=0)
        with col3:
            home          = st.selectbox("🏠 Home & Hobbies",
                                         list(CD_OPTIONS.keys()), index=0)
            care          = st.selectbox("🪥 Personal Care",
                                         list(CD_OPTIONS_CARE.keys()), index=0)

        # Read numeric values
        M_val  = CD_OPTIONS[memory]
        O_val  = CD_OPTIONS[orientation]
        J_val  = CD_OPTIONS[judgment]
        C_val  = CD_OPTIONS[communication]
        H_val  = CD_OPTIONS[home]
        PC_val = CD_OPTIONS_CARE[care]

        # Live CDR-SB preview
        sb_live = compute_cdr_sb(M_val, O_val, J_val, C_val, H_val, PC_val)

        # Staging label for CDR-SB
        if sb_live == 0:
            sb_stage = "Normal (CDR 0)"
        elif sb_live <= 4.0:
            sb_stage = "Very Mild Dementia range (CDR 0.5)"
        elif sb_live <= 9.0:
            sb_stage = "Mild Dementia range (CDR 1)"
        elif sb_live <= 15.5:
            sb_stage = "Moderate Dementia range (CDR 2)"
        else:
            sb_stage = "Severe Dementia range (CDR 3)"

        
        st.markdown("---")
        if st.button("🧪 Run Clinical Rule Engine", type="primary", use_container_width=True):
            (tab_pred, tab_conf, tab_probs,
             global_cdr, cdr_sb, reasoning) = predict_clinical(
                M_val, O_val, J_val, C_val, H_val, PC_val, genotype)

            st.session_state.update({
                "tab_pred":  tab_pred,
                "tab_conf":  tab_conf,
                "tab_probs": tab_probs,
            })

            
            # ── Result card ─────────────────────────────────────────────────
            st.markdown(f"""
            <div class='result-box'>
                <div class='prediction-label'>Clinical Rule Engine Diagnosis:</div>
                <div class='prediction-value'>{tab_pred}</div>
                <div class='confidence-text'>
                    Confidence: {tab_conf:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Class Probabilities (after APOE adjustment)")
            for lbl, p in sorted(tab_probs.items(), key=lambda x: -x[1]):
                prob_bar(lbl, p, SEVERITY_COLORS[lbl])

            interpretation_block(tab_pred)

    # ── Tab 3: Fusion ─────────────────────────────────────────────────────────
    with tab_results:
        st.header("📊 Multimodal Fusion Results")

        img_probs_stored = st.session_state.get("img_probs")
        tab_probs_stored = st.session_state.get("tab_probs")
        have_img = img_probs_stored is not None
        have_tab = tab_probs_stored is not None

        if not have_img and not have_tab:
            st.info("Run at least one model (Image or Clinical) to see results here.")
            return

        col_img, col_tab = st.columns(2)
        with col_img:
            st.subheader("🖼️ Image Model")
            if have_img:
                st.markdown(f"""
                <div class='result-box'>
                    <div class='prediction-label'>Diagnosis:</div>
                    <div class='prediction-value'>{st.session_state["img_pred"]}</div>
                    <div class='confidence-text'>Confidence: {st.session_state["img_conf"]:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                for lbl, p in sorted(img_probs_stored.items(), key=lambda x: -x[1]):
                    prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            else:
                st.info("No image analysis run yet.")

        with col_tab:
            st.subheader("📋 Clinical Rule Engine")
            if have_tab:
                st.markdown(f"""
                <div class='result-box'>
                    <div class='prediction-label'>Diagnosis:</div>
                    <div class='prediction-value'>{st.session_state["tab_pred"]}</div>
                    <div class='confidence-text'>Confidence: {st.session_state["tab_conf"]:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                for lbl, p in sorted(tab_probs_stored.items(), key=lambda x: -x[1]):
                    prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            else:
                st.info("No clinical analysis run yet.")

        if have_img and have_tab:
            st.markdown("---")
            st.subheader("🔀 Fused Prediction (Weighted Average)")
            fused_label, fused_conf, fused_probs = fuse_predictions(
                img_probs_stored, tab_probs_stored, img_w, tab_w)
            st.markdown(f"""
            <div class='fusion-box'>
                <div class='prediction-label'>Final Fused Diagnosis:</div>
                <div class='prediction-value'>{fused_label}</div>
                <div class='confidence-text'>
                    Fused Score: {fused_conf:.1f}%
                    &nbsp;|&nbsp; Image weight: {img_w}
                    &nbsp;|&nbsp; Clinical weight: {tab_w}
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("#### Fused Probabilities")
            for lbl, p in sorted(fused_probs.items(), key=lambda x: -x[1]):
                prob_bar(lbl, p, SEVERITY_COLORS[lbl])
            st.markdown("### 💡 Interpretation")
            interpretation_block(fused_label)

        elif have_img:
            st.markdown("### 💡 Interpretation (Image model only)")
            interpretation_block(st.session_state["img_pred"])
        else:
            st.markdown("### 💡 Interpretation (Clinical rule engine only)")
            interpretation_block(st.session_state["tab_pred"])


if __name__ == "__main__":
    main()