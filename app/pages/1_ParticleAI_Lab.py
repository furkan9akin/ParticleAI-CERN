from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "gradient_boosting_v1.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="ParticleAI Lab",
    page_icon="🔬",
    layout="wide"
)

st.title("ParticleAI Lab")

st.write(
    "Explore photon data using two different input methods."
)

st.info(
    "This is an educational machine learning tool. "
    "Predictions are not confirmed Higgs boson detections."
)

if MODEL_PATH.exists():
    try:
        model = load_model()
        st.success("Gradient Boosting model loaded successfully.")
    except Exception:
        model = None
        st.error("The saved model could not be loaded.")
else:
    model = None
    st.warning("Saved model file not found.")

# --------------------------------------------------
# INPUT TABS
# --------------------------------------------------

csv_tab, manual_tab = st.tabs([
    "Upload CSV",
    "Enter Photon Values"
])


# --------------------------------------------------
# CSV DATA ANALYSIS
# --------------------------------------------------

with csv_tab:

    st.subheader("CSV Data Analysis")

    st.write(
        "Upload a CSV file containing photon kinematic features."
    )

    st.caption(
        "Maximum supported file size: 10 MB. "
        "Photon transverse momenta must be in GeV."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        if uploaded_file.size > 10 * 1024 * 1024:

            st.error(
                "Please upload a CSV file smaller than 10 MB."
            )

        else:

            try:
                df = pd.read_csv(uploaded_file)

            except Exception:

                st.error(
                    "The CSV file could not be read. "
                    "Please check its format."
                )

            else:

                if df.empty:
                    st.error(
                        "The CSV file contains no events. "
                        "Please upload a file with at least one data row."
                    )
                else:
                    st.success("CSV file loaded successfully!")

                st.write("Number of events:", len(df))
                st.write("Number of columns:", len(df.columns))

                st.subheader("Data Preview")
                st.dataframe(df.head(5))

                # ----------------------------------
                # REQUIRED FEATURES
                # ----------------------------------

                required_features = [
                    "lead_photon_pt_GeV",
                    "sublead_photon_pt_GeV",
                    "lead_photon_eta",
                    "sublead_photon_eta",
                    "delta_eta_gg",
                    "delta_phi_gg",
                    "diphoton_pt_GeV"
                ]

                missing_features = [
                    feature
                    for feature in required_features
                    if feature not in df.columns
                ]

                st.subheader("Feature Validation")

                if missing_features:

                    st.error(
                        "Missing required features: "
                        + ", ".join(missing_features)
                    )

                else:

                    st.success(
                        "All 7 required photon features are present."
                    )

                    # ----------------------------------
                    # DATA QUALITY CHECK
                    # ----------------------------------

                    numeric_data = df[required_features].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    has_missing = numeric_data.isna().any().any()

                    has_infinite = not np.isfinite(
                        numeric_data.to_numpy(dtype=float)
                    ).all()

                    st.subheader("Data Quality Check")

                    if len(df) == 0:

                        st.error(
                            "The CSV file contains no events."
                        )

                    elif has_missing:

                        st.error(
                            "Some required features contain missing "
                            "or non-numeric values."
                        )

                    elif has_infinite:

                        st.error(
                            "Some required features contain "
                            "infinite values."
                        )

                    else:

                        st.success(
                            "Data quality check passed: all required "
                            "features contain finite numeric values."
                        )

                        # ----------------------------------
                        # PHOTON PT CHECK
                        # ----------------------------------

                        st.subheader(
                            "Physical Consistency Check"
                        )

                        lead_pt = numeric_data[
                            "lead_photon_pt_GeV"
                        ]

                        sublead_pt = numeric_data[
                            "sublead_photon_pt_GeV"
                        ]

                        invalid_pt = (
                            (lead_pt <= 0)
                            | (sublead_pt <= 0)
                            | (lead_pt < sublead_pt)
                        )

                        invalid_count = int(
                            invalid_pt.sum()
                        )

                        if invalid_count > 0:

                            st.error(
                                f"{invalid_count} events have invalid "
                                "or incorrectly ordered photon pT values."
                            )

                        else:

                            st.success(
                                "Photon pT check passed: all values are "
                                "positive and correctly ordered."
                            )

                        # ----------------------------------
                        # DELTA ETA CHECK
                        # ----------------------------------

                        st.subheader(
                            "Delta Eta Consistency"
                        )

                        calculated_delta_eta = (
                            numeric_data["lead_photon_eta"]
                            - numeric_data["sublead_photon_eta"]
                        )

                        provided_delta_eta = numeric_data[
                            "delta_eta_gg"
                        ]

                        matching_delta_eta = np.isclose(
                            calculated_delta_eta,
                            provided_delta_eta,
                            atol=0.001,
                            rtol=0
                        )

                        inconsistent_eta_count = int(
                            (~matching_delta_eta).sum()
                        )

                        if inconsistent_eta_count > 0:

                            st.error(
                                f"{inconsistent_eta_count} events have "
                                "inconsistent delta eta values."
                            )

                        else:

                            st.success(
                                "Delta eta check passed: values match "
                                "the photon eta differences."
                            )

                        # ----------------------------------
                        # DELTA PHI RANGE CHECK
                        # ----------------------------------

                        st.subheader(
                            "Delta Phi Range Check"
                        )

                        delta_phi = numeric_data[
                            "delta_phi_gg"
                        ]

                        invalid_phi = (
                            (delta_phi < -np.pi - 0.001)
                            | (delta_phi > np.pi + 0.001)
                        )

                        invalid_phi_count = int(
                            invalid_phi.sum()
                        )

                        if invalid_phi_count > 0:

                            st.error(
                                f"{invalid_phi_count} events have "
                                "delta phi values outside [-pi, pi]."
                            )

                        else:

                            st.success(
                                "Delta phi check passed: all angular "
                                "differences are within [-pi, pi]."
                            )

                        # ----------------------------------
                        # DIPHOTON PT CONSISTENCY
                        # ----------------------------------

                        st.subheader(
                            "Diphoton pT Consistency"
                        )

                        calculated_diphoton_pt = np.sqrt(
                            np.maximum(
                                lead_pt**2
                                + sublead_pt**2
                                + 2 * lead_pt * sublead_pt
                                * np.cos(delta_phi),
                                0
                            )
                        )

                        provided_diphoton_pt = numeric_data[
                            "diphoton_pt_GeV"
                        ]

                        matching_diphoton_pt = np.isclose(
                            calculated_diphoton_pt,
                            provided_diphoton_pt,
                            atol=0.01,
                            rtol=0.0001
                        )

                        inconsistent_diphoton_count = int(
                            (~matching_diphoton_pt).sum()
                        )

                        if inconsistent_diphoton_count > 0:

                            st.error(
                                f"{inconsistent_diphoton_count} events "
                                "have inconsistent diphoton pT values."
                            )

                        else:

                            st.success(
                                "Diphoton pT check passed: values match "
                                "the photon momenta and angular separation."
                            )

                        st.caption(
                            "These checks test numerical and internal "
                            "consistency. They do not prove that the "
                            "uploaded data comes from ATLAS or that "
                            "the physical units are correct."
                        )

                        # ----------------------------------
                        # MODEL PREDICTIONS
                        # ----------------------------------

                        if (
                            model is not None
                            and invalid_count == 0
                            and inconsistent_eta_count == 0
                            and invalid_phi_count == 0
                            and inconsistent_diphoton_count == 0
                        ):
                            st.subheader("Model Predictions")

                            st.write(
                                "The model can generate signal-likeness "
                                "scores for the uploaded events."
                            )

                            if st.button("Analyze Uploaded Events"):
                                X = numeric_data[required_features]

                                scores = model.predict_proba(X)[:, 1]

                                results = pd.DataFrame({
                                    "Event": np.arange(1, len(scores) + 1),
                                    "Signal-likeness Score": scores
                                })

                                st.success(
                                    f"Analyzed {len(results)} events."
                                )

                                st.dataframe(
                                    results.head(10),
                                    hide_index=True
                                )

                                st.download_button(
                                    label="Download Analysis Results (CSV)",
                                    data=results.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name="particleai_analysis_results.csv",
                                    mime="text/csv"
                                )
                                                                # Create a CSV analysis summary report
                                mean_score = float(np.mean(scores))

                                high_score_count = int(
                                    (scores >= 0.5).sum()
                                )

                                csv_report_lines = [
                                    "# ParticleAI Lab - CSV Analysis Report",
                                    "",
                                    "## Dataset Summary",
                                    "",
                                    f"- Events analyzed: {len(scores)}",
                                    "- Model: Histogram-based Gradient Boosting",
                                    "- Classification threshold: 0.5",
                                    "",
                                    "## Model Score Summary",
                                    "",
                                    f"- Mean signal-likeness score: {mean_score:.4f}",
                                    f"- Events with score >= 0.5: {high_score_count}",
                                    "",
                                    "## Scientific Limitations",
                                    "",
                                    "This is an educational machine learning analysis.",
                                    "The model was trained on balanced, unweighted",
                                    "Monte Carlo signal and background samples.",
                                    "Model scores are not calibrated Higgs probabilities.",
                                    "High-scoring events are not confirmed Higgs events.",
                                    "Passing data checks does not verify the data source",
                                    "or prove compatibility with the training simulation.",
                                    "Classification accuracy cannot be calculated",
                                    "without reliable ground-truth labels.",
                                    ""
                                ]

                                csv_report_text = "\n".join(
                                    csv_report_lines
                                )

                                st.download_button(
                                    label="Download CSV Analysis Report",
                                    data=csv_report_text,
                                    file_name="particleai_csv_report.md",
                                    mime="text/markdown",
                                    on_click="ignore"
                                )

                                st.caption(
                                    "Only the first 10 results are shown. "
                                    "Scores are not calibrated Higgs "
                                    "probabilities or confirmed detections."
                                )

                        else:
                            st.warning(
                                "Model analysis is unavailable until "
                                "all data checks pass."
                            )



# --------------------------------------------------
# MANUAL PHOTON INPUT
# --------------------------------------------------



with manual_tab:

    st.subheader("Manual Photon Input")

    st.write(
        "Enter the properties of two photons. "
        "Transverse momentum must be in GeV "
        "and angles must be in radians."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Photon 1")

        pt1 = st.number_input(
            "Photon 1 pT (GeV)",
            min_value=0.01,
            value=60.0,
            step=1.0
        )

        eta1 = st.number_input(
            "Photon 1 eta",
            value=0.4,
            step=0.1
        )

        phi1 = st.number_input(
            "Photon 1 phi (radians)",
            min_value=-float(np.pi),
            max_value=float(np.pi),
            value=0.0,
            step=0.1
        )

    with col2:
        st.markdown("### Photon 2")

        pt2 = st.number_input(
            "Photon 2 pT (GeV)",
            min_value=0.01,
            value=45.0,
            step=1.0
        )

        eta2 = st.number_input(
            "Photon 2 eta",
            value=-0.3,
            step=0.1
        )

        phi2 = st.number_input(
            "Photon 2 phi (radians)",
            min_value=-float(np.pi),
            max_value=float(np.pi),
            value=2.8,
            step=0.1
        )

    if st.button("Calculate Photon Features"):

        # Sort photons by transverse momentum
        if pt2 > pt1:
            pt1, pt2 = pt2, pt1
            eta1, eta2 = eta2, eta1
            phi1, phi2 = phi2, phi1

        # Angular differences
        delta_eta = eta1 - eta2

        delta_phi = (
            (phi1 - phi2 + np.pi) % (2 * np.pi)
        ) - np.pi

        # Diphoton transverse momentum
        diphoton_pt = np.sqrt(
            max(
                pt1**2
                + pt2**2
                + 2 * pt1 * pt2 * np.cos(delta_phi),
                0
            )
        )

        # Diphoton invariant mass
        mass_squared = (
            2 * pt1 * pt2
            * (np.cosh(delta_eta) - np.cos(delta_phi))
        )

        diphoton_mass = np.sqrt(
            max(mass_squared, 0)
        )

        # Same seven features used during model training
        features = pd.DataFrame([{
            "lead_photon_pt_GeV": pt1,
            "sublead_photon_pt_GeV": pt2,
            "lead_photon_eta": eta1,
            "sublead_photon_eta": eta2,
            "delta_eta_gg": delta_eta,
            "delta_phi_gg": delta_phi,
            "diphoton_pt_GeV": diphoton_pt
        }])

        st.success("Photon features calculated successfully!")

        st.subheader("Calculated Features")

        st.dataframe(
            features,
            hide_index=True
        )

        st.metric(
            "Diphoton Invariant Mass",
            f"{diphoton_mass:.2f} GeV"
        )

        st.caption(
            "Invariant mass is calculated for exploration "
            "but is not one of the seven model input features. "
            "These manually entered values are not verified "
            "ATLAS collision events."
        )
                # Predict using the saved Gradient Boosting model
        st.subheader("Machine Learning Prediction")

        if model is not None:

            signal_score = float(
                model.predict_proba(features)[0, 1]
            )

            st.metric(
                "Signal-likeness Score",
                f"{signal_score:.4f}"
            )

            if signal_score >= 0.5:
                st.write(
                    "The model classifies this example as "
                    "signal-like using the 0.5 threshold."
                )
            else:
                st.write(
                    "The model classifies this example as "
                    "background-like using the 0.5 threshold."
                )

            st.warning(
                "This is a model score, not a calibrated "
                "probability of a Higgs boson. Manually "
                "entered values may differ from the data "
                "used to train the model."
            )
                        # Create a downloadable Markdown report
            classification = (
                "Signal-like"
                if signal_score >= 0.5
                else "Background-like"
            )

            report_lines = [
                "# ParticleAI Lab - Manual Analysis Report",
                "",
                "## Analysis Results",
                "",
                f"- Diphoton invariant mass: {diphoton_mass:.2f} GeV",
                f"- Signal-likeness score: {signal_score:.4f}",
                f"- Model classification: {classification}",
                "- Classification threshold: 0.5",
                "",
                "## Calculated Photon Features",
                ""
            ]

            for feature_name, feature_value in features.iloc[0].items():
                report_lines.append(
                    f"- {feature_name}: {feature_value:.4f}"
                )

            report_lines.extend([
                "",
                "## Scientific Limitations",
                "",
                "This is an educational machine learning analysis.",
                "The photon values were entered manually.",
                "They are not verified ATLAS collision events.",
                "The model was trained on balanced Monte Carlo samples.",
                "The score is not a calibrated Higgs probability.",
                "The classification does not confirm a Higgs boson.",
                ""
            ])

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Manual Analysis Report",
                data=report_text,
                file_name="particleai_manual_report.md",
                mime="text/markdown",
                on_click="ignore"
            )

        else:
            st.error(
                "The saved model is not available."
            )