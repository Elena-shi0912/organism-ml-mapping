from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLSI_PATH, INPUT_CODE_COL, INPUT_NAME_COL, TARGET_COLS
from src.preprocessing import load_and_clean_clsi
from src.retrieval_baseline import build_retrieval_index, retrieve_topk
from src.embedding_retrieval import build_lsa_index, retrieve_lsa_topk


MODELS_DIR = ROOT / "models"
FEEDBACK_DIR = ROOT / "data" / "feedback"
FEEDBACK_PATH = FEEDBACK_DIR / "feedback_log.csv"

TARGET_TO_FILENAME = {
    "Reported Name": "Reported_Name",
    "CLSI Breakpoint Group": "CLSI_Breakpoint_Group",
    "Bacteria Gram Stain": "Bacteria_Gram_Stain",
}

SHOW_SUPERVISED_MODELS = os.getenv("SHOW_SUPERVISED_MODELS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

MODEL_OPTIONS = {
    "Retrieval (Top-1)": "retrieval_top1",
    "Retrieval (Top-3)": "retrieval_top3",
    "LSA Retrieval (Top-1)": "lsa_retrieval_top1",
    "LSA Retrieval (Top-3)": "lsa_retrieval_top3",
}

if SHOW_SUPERVISED_MODELS:
    MODEL_OPTIONS = {
        "Logistic Regression": "logreg",
        "Random Forest": "random_forest",
        **MODEL_OPTIONS,
    }

INPUT_OPTIONS = {
    "Code only": "code_only",
    "Full name only": "name_only",
    "Code + full name": "code_plus_name",
}


@st.cache_resource
def load_target_model(model_key: str, input_variant: str, target_name: str):
    filename_target = TARGET_TO_FILENAME[target_name]
    model_path = MODELS_DIR / f"{model_key}_{input_variant}_{filename_target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Make sure you trained and saved this model first."
        )
    return joblib.load(model_path)


@st.cache_resource
def load_models(model_key: str, input_variant: str) -> Dict[str, object]:
    return {
        target: load_target_model(model_key, input_variant, target)
        for target in TARGET_TO_FILENAME
    }


@st.cache_resource
def load_retrieval_index(input_variant: str):
    clsi_df = load_and_clean_clsi(CLSI_PATH)
    return build_retrieval_index(
        full_clsi_df=clsi_df,
        input_variant=input_variant,
        normalize=True,
    )


@st.cache_resource
def load_lsa_index(input_variant: str):
    clsi_df = load_and_clean_clsi(CLSI_PATH)
    return build_lsa_index(
        index_df=clsi_df,
        input_variant=input_variant,
        normalize=True,
    )


def build_input_text(input_variant: str, species_code: str, species_name: str) -> str:
    species_code = (species_code or "").strip()
    species_name = (species_name or "").strip()

    if input_variant == "code_only":
        return f"species_code={species_code}"
    if input_variant == "name_only":
        return f"organism={species_name}"
    if input_variant == "code_plus_name":
        return f"species_code={species_code} organism={species_name}"
    raise ValueError(f"Unknown input_variant: {input_variant}")


def predict_all(models: Dict[str, object], input_text: str) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for target, model in models.items():
        pred = model.predict([input_text])[0]

        confidence = None
        top_candidates: List[str] = [str(pred)]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([input_text])[0]
            classes = model.classes_
            top_idx = proba.argsort()[::-1][:3]
            top_candidates = [f"{classes[i]} ({proba[i]:.3f})" for i in top_idx]
            confidence = float(proba[top_idx[0]])

        results[target] = {
            "prediction": pred,
            "confidence": confidence,
            "top_candidates": top_candidates,
        }

    return results


def predict_all_retrieval(
    index,
    species_code: str,
    species_name: str,
    k: int = 1,
) -> Dict[str, Dict[str, object]]:
    input_df = pd.DataFrame({
        INPUT_CODE_COL: [species_code],
        INPUT_NAME_COL: [species_name],
    })
    candidates = retrieve_topk(index, input_df, k=max(3, k))

    results: Dict[str, Dict[str, object]] = {}
    for target in TARGET_COLS:
        row_candidates = candidates[target][0]
        top1_label, top1_score = row_candidates[0]
        topk_labels = [f"{label} ({score:.3f})" for label, score in row_candidates[:max(3, k)]]

        if k == 1:
            prediction = top1_label
        else:
            labels_only = [label for label, _ in row_candidates[:k]]
            counts = {}
            for item in labels_only:
                counts[item] = counts.get(item, 0) + 1
            max_count = max(counts.values())
            tied = [label for label, count in counts.items() if count == max_count]
            prediction = top1_label if top1_label in tied else tied[0]

        results[target] = {
            "prediction": prediction,
            "confidence": float(top1_score),
            "top_candidates": topk_labels,
        }

    return results


def predict_all_lsa_retrieval(
    index,
    species_code: str,
    species_name: str,
    k: int = 1,
) -> Dict[str, Dict[str, object]]:
    input_df = pd.DataFrame({
        INPUT_CODE_COL: [species_code],
        INPUT_NAME_COL: [species_name],
    })
    candidates = retrieve_lsa_topk(index, input_df, k=max(3, k))

    results: Dict[str, Dict[str, object]] = {}
    for target in TARGET_COLS:
        row_candidates = candidates[target][0]
        top1_label, top1_score = row_candidates[0]
        topk_labels = [f"{label} ({score:.3f})" for label, score in row_candidates[:max(3, k)]]

        if k == 1:
            prediction = top1_label
        else:
            labels_only = [label for label, _ in row_candidates[:k]]
            counts = {}
            for item in labels_only:
                counts[item] = counts.get(item, 0) + 1
            max_count = max(counts.values())
            tied = [label for label, count in counts.items() if count == max_count]
            prediction = top1_label if top1_label in tied else tied[0]

        results[target] = {
            "prediction": prediction,
            "confidence": float(top1_score),
            "top_candidates": topk_labels,
        }

    return results


def ensure_feedback_file() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_PATH.exists():
        pd.DataFrame(
            columns=[
                "model_type",
                "input_variant",
                "species_code",
                "species_full_name",
                "input_text",
                "pred_reported_name",
                "pred_breakpoint_group",
                "pred_gram_stain",
                "confidence_reported_name",
                "confidence_breakpoint_group",
                "confidence_gram_stain",
                "decision",
                "corrected_reported_name",
                "corrected_breakpoint_group",
                "corrected_gram_stain",
                "notes",
                "timestamp",
            ]
        ).to_csv(FEEDBACK_PATH, index=False)


def save_feedback(
    model_label: str,
    input_label: str,
    species_code: str,
    species_name: str,
    input_text: str,
    results: Dict[str, Dict[str, object]],
    decision: str,
    corrected_reported_name: str,
    corrected_breakpoint_group: str,
    corrected_gram_stain: str,
    notes: str,
) -> None:
    ensure_feedback_file()

    row = {
        "model_type": model_label,
        "input_variant": input_label,
        "species_code": species_code.strip(),
        "species_full_name": species_name.strip(),
        "input_text": input_text,
        "pred_reported_name": results["Reported Name"]["prediction"],
        "pred_breakpoint_group": results["CLSI Breakpoint Group"]["prediction"],
        "pred_gram_stain": results["Bacteria Gram Stain"]["prediction"],
        "confidence_reported_name": results["Reported Name"]["confidence"],
        "confidence_breakpoint_group": results["CLSI Breakpoint Group"]["confidence"],
        "confidence_gram_stain": results["Bacteria Gram Stain"]["confidence"],
        "decision": decision,
        "corrected_reported_name": corrected_reported_name.strip(),
        "corrected_breakpoint_group": corrected_breakpoint_group.strip(),
        "corrected_gram_stain": corrected_gram_stain.strip(),
        "notes": notes.strip(),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }

    pd.DataFrame([row]).to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)


def init_session_state() -> None:
    defaults = {
        "prediction_done": False,
        "results": None,
        "input_text": "",
        "saved_model_label": None,
        "saved_input_label": None,
        "saved_species_code": "",
        "saved_species_name": "",
        "save_success_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_prediction_state() -> None:
    st.session_state.prediction_done = False
    st.session_state.results = None
    st.session_state.input_text = ""
    st.session_state.saved_model_label = None
    st.session_state.saved_input_label = None
    st.session_state.saved_species_code = ""
    st.session_state.saved_species_name = ""
    st.session_state.save_success_message = ""

    keys_to_remove = [
        "review_decision",
        "notes",
        "Reported Name_mode",
        "CLSI Breakpoint Group_mode",
        "Bacteria Gram Stain_mode",
        "Reported Name_select",
        "CLSI Breakpoint Group_select",
        "Bacteria Gram Stain_select",
        "Reported Name_manual",
        "CLSI Breakpoint Group_manual",
        "Bacteria Gram Stain_manual",
    ]

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


def parse_candidate_label(candidate: str) -> str:
    return candidate.split(" (")[0].strip() if " (" in candidate else candidate.strip()


def get_deduped_candidates(results: Dict[str, Dict[str, object]], target: str) -> List[str]:
    prediction = str(results[target]["prediction"])
    top_candidates = [parse_candidate_label(x) for x in results[target]["top_candidates"]]

    deduped_candidates = []
    for item in [prediction] + top_candidates:
        if item and item not in deduped_candidates:
            deduped_candidates.append(item)

    return deduped_candidates


def render_prediction_cards(results: Dict[str, Dict[str, object]]) -> None:
    st.subheader("Predictions")

    pred_cols = st.columns(3)
    ordered_targets = [
        "Reported Name",
        "CLSI Breakpoint Group",
        "Bacteria Gram Stain",
    ]

    for ui_col, target in zip(pred_cols, ordered_targets):
        with ui_col:
            st.markdown(f"**{target}**")
            st.write(results[target]["prediction"])

            if results[target]["confidence"] is not None:
                st.caption(f"Confidence: {results[target]['confidence']:.3f}")

            with st.expander("Top candidates"):
                for item in results[target]["top_candidates"]:
                    st.write(item)


def render_correction_widget(results: Dict[str, Dict[str, object]], target: str, label: str) -> str:
    prediction = str(results[target]["prediction"])
    deduped_candidates = get_deduped_candidates(results, target)

    mode_key = f"{target}_mode"
    select_key = f"{target}_select"
    manual_key = f"{target}_manual"

    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Choose from top candidates"
    if select_key not in st.session_state:
        st.session_state[select_key] = prediction
    if manual_key not in st.session_state:
        st.session_state[manual_key] = prediction

    mode = st.radio(
        f"{label} correction mode",
        ["Choose from top candidates", "Enter manually"],
        horizontal=True,
        key=mode_key,
    )

    if mode == "Choose from top candidates":
        if st.session_state[select_key] not in deduped_candidates:
            st.session_state[select_key] = deduped_candidates[0]

        return st.selectbox(
            f"Corrected {label}",
            deduped_candidates,
            key=select_key,
        )

    return st.text_input(
        f"Corrected {label}",
        key=manual_key,
    )


def valid_input(input_variant: str, species_code: str, species_name: str) -> bool:
    species_code = species_code.strip()
    species_name = species_name.strip()

    if input_variant == "code_only":
        return bool(species_code)
    if input_variant == "name_only":
        return bool(species_name)
    if input_variant == "code_plus_name":
        return bool(species_code or species_name)
    return False


st.set_page_config(page_title="CLSI Human-in-the-Loop", layout="wide")
init_session_state()

st.title("CLSI Human-in-the-Loop Review")
st.caption("Predict Reported Name, CLSI Breakpoint Group, and Gram Stain from species code and/or full name.")

with st.sidebar:
    st.header("Model settings")
    model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()), index=0)
    input_label = st.selectbox("Choose input type", list(INPUT_OPTIONS.keys()), index=2)

    if st.button("Start new review"):
        clear_prediction_state()
        st.rerun()

model_key = MODEL_OPTIONS[model_label]
input_variant = INPUT_OPTIONS[input_label]

col1, col2 = st.columns(2)
with col1:
    species_code = st.text_input(
        "Species Code",
        placeholder="e.g. ECO",
        value=st.session_state.saved_species_code if st.session_state.prediction_done else "",
    )
with col2:
    species_name = st.text_input(
        "Species Full Name",
        placeholder="e.g. Escherichia coli",
        value=st.session_state.saved_species_name if st.session_state.prediction_done else "",
    )

if input_variant == "code_only" and not species_code.strip():
    st.info("Enter a species code to run prediction.")
if input_variant == "name_only" and not species_name.strip():
    st.info("Enter a species full name to run prediction.")
if input_variant == "code_plus_name" and not (species_code.strip() or species_name.strip()):
    st.info("Enter species code, species full name, or both to run prediction.")

run_prediction = st.button("Predict", type="primary")

if run_prediction:
    if not valid_input(input_variant, species_code, species_name):
        st.error("Please provide the required input(s) before running prediction.")
    else:
        try:
            if model_key.startswith("retrieval_"):
                k = 1 if model_key == "retrieval_top1" else 3
                index = load_retrieval_index(input_variant)
                input_text = build_input_text(input_variant, species_code, species_name)
                results = predict_all_retrieval(
                    index=index,
                    species_code=species_code,
                    species_name=species_name,
                    k=k,
                )
            elif model_key.startswith("lsa_retrieval_"):
                k = 1 if model_key == "lsa_retrieval_top1" else 3
                index = load_lsa_index(input_variant)
                input_text = build_input_text(input_variant, species_code, species_name)
                results = predict_all_lsa_retrieval(
                    index=index,
                    species_code=species_code,
                    species_name=species_name,
                    k=k,
                )
            else:
                models = load_models(model_key, input_variant)
                input_text = build_input_text(input_variant, species_code, species_name)
                results = predict_all(models, input_text)

            st.session_state.prediction_done = True
            st.session_state.results = results
            st.session_state.input_text = input_text
            st.session_state.saved_model_label = model_label
            st.session_state.saved_input_label = input_label
            st.session_state.saved_species_code = species_code
            st.session_state.saved_species_name = species_name
            st.session_state.save_success_message = ""

            for key in [
                "review_decision",
                "notes",
                "Reported Name_mode",
                "CLSI Breakpoint Group_mode",
                "Bacteria Gram Stain_mode",
                "Reported Name_select",
                "CLSI Breakpoint Group_select",
                "Bacteria Gram Stain_select",
                "Reported Name_manual",
                "CLSI Breakpoint Group_manual",
                "Bacteria Gram Stain_manual",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

        except Exception as exc:
            st.error(str(exc))
            st.session_state.prediction_done = False
            st.session_state.results = None

if st.session_state.prediction_done and st.session_state.results is not None:
    results = st.session_state.results
    input_text = st.session_state.input_text

    render_prediction_cards(results)

    st.divider()
    st.subheader("Review decision")

    decision = st.radio(
        "Choose one",
        [
            "Accept",
            "Decline with correction",
            "Flag as uncertain",
        ],
        horizontal=True,
        key="review_decision",
    )

    corrected_reported_name = str(results["Reported Name"]["prediction"])
    corrected_breakpoint_group = str(results["CLSI Breakpoint Group"]["prediction"])
    corrected_gram_stain = str(results["Bacteria Gram Stain"]["prediction"])

    if decision == "Decline with correction":
        st.markdown("**Provide corrected labels**")

        c1, c2, c3 = st.columns(3)

        with c1:
            corrected_reported_name = render_correction_widget(results, "Reported Name", "Reported Name")
        with c2:
            corrected_breakpoint_group = render_correction_widget(
                results,
                "CLSI Breakpoint Group",
                "Breakpoint Group",
            )
        with c3:
            corrected_gram_stain = render_correction_widget(
                results,
                "Bacteria Gram Stain",
                "Gram Stain",
            )

    notes = st.text_area(
        "Optional notes",
        placeholder="Add context or why this case is uncertain.",
        key="notes",
    )

    if st.button("Save feedback"):
        if decision == "Decline with correction":
            original_reported_name = str(results["Reported Name"]["prediction"])
            original_breakpoint_group = str(results["CLSI Breakpoint Group"]["prediction"])
            original_gram_stain = str(results["Bacteria Gram Stain"]["prediction"])

            nothing_changed = (
                corrected_reported_name.strip() == original_reported_name.strip()
                and corrected_breakpoint_group.strip() == original_breakpoint_group.strip()
                and corrected_gram_stain.strip() == original_gram_stain.strip()
            )

            if nothing_changed:
                st.error(
                    "You selected 'Decline with correction', but no correction was made. "
                    "Please change at least one field or choose a different review decision."
                )
            else:
                save_feedback(
                    model_label=st.session_state.saved_model_label,
                    input_label=st.session_state.saved_input_label,
                    species_code=st.session_state.saved_species_code,
                    species_name=st.session_state.saved_species_name,
                    input_text=input_text,
                    results=results,
                    decision=decision,
                    corrected_reported_name=corrected_reported_name,
                    corrected_breakpoint_group=corrected_breakpoint_group,
                    corrected_gram_stain=corrected_gram_stain,
                    notes=notes,
                )
                st.session_state.save_success_message = f"Saved to {FEEDBACK_PATH}"
                st.rerun()
        else:
            save_feedback(
                model_label=st.session_state.saved_model_label,
                input_label=st.session_state.saved_input_label,
                species_code=st.session_state.saved_species_code,
                species_name=st.session_state.saved_species_name,
                input_text=input_text,
                results=results,
                decision=decision,
                corrected_reported_name=corrected_reported_name,
                corrected_breakpoint_group=corrected_breakpoint_group,
                corrected_gram_stain=corrected_gram_stain,
                notes=notes,
            )
            st.session_state.save_success_message = f"Saved to {FEEDBACK_PATH}"
            st.rerun()

if st.session_state.save_success_message:
    st.success(st.session_state.save_success_message)

st.divider()
st.subheader("Feedback log")

if FEEDBACK_PATH.exists():
    feedback_df = pd.read_csv(FEEDBACK_PATH)
    st.dataframe(feedback_df.tail(20), use_container_width=True)
else:
    st.caption("No feedback saved yet.")
