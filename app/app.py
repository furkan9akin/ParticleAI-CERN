
import streamlit as st

from pathlib import Path

st.set_page_config(
    page_title="ParticleAI-CERN",
    page_icon="🔬",
    layout="wide"
)

st.title("ParticleAI-CERN")

st.write(
    "An independent high school research project by Furkan Akın."
)

st.info(
    "Exploring machine learning for CERN ATLAS "
    "Higgs signal and diphoton background classification."
)

st.warning(
    "Educational project — not an official ATLAS result."
)
st.divider()

st.header("Final Model Evaluation")

st.write(
    "The selected Gradient Boosting model was evaluated "
    "on 3,000 held-out Monte Carlo test events."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "81.37%")
    st.metric("Precision", "77.60%")

with col2:
    st.metric("Recall", "88.20%")
    st.metric("F1 Score", "82.56%")

with col3:
    st.metric("ROC-AUC", "0.8922")
    st.metric("PR-AUC", "0.8636")

st.caption(
    "Results are based on a balanced simulated dataset. "
    "They do not measure Higgs detection accuracy in real ATLAS data."
)


st.divider()

st.header("Understanding the Model")

st.write(
    "Permutation feature importance measures how much "
    "the model's validation ROC-AUC decreases when "
    "the values of one feature are shuffled."
)

project_root = Path(__file__).resolve().parent.parent

importance_plot = (
    project_root / "figures" / "feature_importance.png"
)

if importance_plot.exists():
    st.image(
        str(importance_plot),
        caption="Permutation feature importance on validation data."
    )
else:
    st.warning("Feature importance figure not found.")

st.caption(
    "Feature importance describes model behavior, "
    "not direct physical causation. Correlated features "
    "can also affect the interpretation."
)

st.divider()

st.header("Simulation vs Real ATLAS Data")

st.write(
    "The trained model was also applied to 63,195 real "
    "ATLAS collision events. The figure compares model "
    "score distributions for simulated Higgs events, "
    "simulated diphoton background, and real collision data."
)

comparison_plot = (
    project_root / "figures" / "mc_vs_real_scores.png"
)

if comparison_plot.exists():
    st.image(
        str(comparison_plot),
        caption="Model score distributions for Monte Carlo and real ATLAS data."
    )
else:
    st.warning("MC vs real data comparison figure not found.")

st.warning(
    "Real ATLAS events do not have known Higgs/background "
    "labels in this dataset. Model scores are not calibrated "
    "Higgs probabilities, and high-scoring events are not "
    "confirmed Higgs boson events."
)

st.divider()

st.header("Machine Learning Model Comparison")

st.write(
    "Three machine learning models were evaluated "
    "using the same validation dataset of 3,000 "
    "simulated events."
)

comparison_data = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [0.7040, 0.7640, 0.8070],
    "Precision": [0.7021, 0.7435, 0.7704],
    "Recall": [0.7087, 0.8060, 0.8747],
    "F1 Score": [0.7054, 0.7735, 0.8192],
    "ROC-AUC": [0.7281, 0.8481, 0.8827]
}

st.dataframe(
    comparison_data,
    hide_index=True,
    width="stretch"
)

st.caption(
    "These are validation results, not final test results. "
    "Gradient Boosting had the highest validation ROC-AUC "
    "among the three tested models."
)

st.divider()

st.header("Scientific Limitations")

st.write(
    "This project is an educational machine learning study "
    "using CERN ATLAS Open Data. It is not an official "
    "ATLAS physics analysis."
)

st.markdown("""
- The full sampled dataset contains 10,000 simulated
  Higgs events and 10,000 simulated background events.
- The dataset was artificially balanced; real Higgs
  events are much rarer.
- Monte Carlo event weights and physical cross-section
  normalization were not applied.
- The signal and background samples have different
  diphoton mass distributions.
- Although invariant mass was excluded from the model
  inputs, other kinematic features contain indirect
  information about mass.
- Only one signal simulation and one background
  simulation were used for classifier training.
- Performance on simulated test events does not establish
  performance on real collision data.
- High model scores do not confirm the presence of
  a Higgs boson.
""")

st.subheader("Data Sources")

st.markdown(
    "- [ATLAS Monte Carlo Open Data]"
    "(https://opendata.cern.ch/record/93922)\n"
    "- [ATLAS Real Collision Data]"
    "(https://opendata.cern.ch/record/93915)\n"
    "- [ParticleAI-CERN GitHub Repository]"
    "(https://github.com/furkan9akin/ParticleAI-CERN)"
)

st.caption(
    "Independent student project by Furkan Akın. "
    "The conclusions are not endorsed by the ATLAS Collaboration."
)