"""
MITIGATION TECHNIQUES
by Esmeralda Ruiz Pujadas
- Post-mitigation fairness mitigation 

The post-mitigation methods are applied AFTER the federated ML
has been aggregated by the server.

The validation data are used to FIT the post-mitigation.

The test data are used only to APPLY the already fitted post-mitigation
and calculate the final results.

Supported methods:
    - eqodds
    - calibrated_eqodds
    - reject_option

Multiple protected attributes can be stored in the same AIF360
BinaryLabelDataset, for example:

    Sex
    Ethnicity

"""

import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset

from aif360.algorithms.postprocessing import (
    EqOddsPostprocessing,
    RejectOptionClassification,
)

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)
import numpy as np
import pandas as pd
from flcore.metrics import calculate_metrics, getFairnessResults

# -------------------------------------------------------------------------
# AIF360 DATASET CREATION
# -------------------------------------------------------------------------

def create_aif360_dataset(
    y_true,
    y_pred,
    protected_data,
    fairness_attribs,
    scores=None,
):
    """
    Create an AIF360 BinaryLabelDataset.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    y_pred : array-like
        Predicted binary labels.

    protected_data : pandas.DataFrame
        DataFrame containing the protected attributes.

        Example:

            Sex | Ethnicity
            1   | 0
            0   | 0
            1   | 1

    fairness_attribs : list
        Names of the protected attributes.

        Example:

            ["Sex", "Ethnicity"]

    scores : array-like, optional
        Prediction scores/probabilities.

        Required by some post-mitigation algorithms such as
        CalibratedEqOddsPostprocessing and RejectOptionClassification.

    Returns
    -------
    BinaryLabelDataset
        AIF360 dataset.
    """

    # Make sure protected_data is a DataFrame.
    if not isinstance(protected_data, pd.DataFrame):
        protected_data = pd.DataFrame(protected_data)

    # Create a clean DataFrame containing:
    #
    #   true label
    #   predicted label
    #   protected attributes
    #
    data = pd.DataFrame()

    data["label"] = np.asarray(y_true).astype(float)

    data["prediction"] = np.asarray(y_pred).astype(float)

    # Add every protected attribute separately.
    #
    # IMPORTANT:
    # We do NOT create something like:
    #
    #     Sex_Ethnicity
    #
    # Instead AIF360 receives both protected attributes.
    for attribute in fairness_attribs:
        data[attribute] = (
            protected_data[attribute]
            .reset_index(drop=True)
            .astype(float)
        )

    # Add prediction scores when supplied.
    if scores is not None:
        data["score"] = np.asarray(scores).astype(float)

    # AIF360 expects the labels to be explicitly defined.
    dataset = BinaryLabelDataset(
        favorable_label=1.0,
        unfavorable_label=0.0,
        df=data,
        label_names=["label"],
        protected_attribute_names=fairness_attribs,
    )

    return dataset


# -------------------------------------------------------------------------
# GROUP DEFINITIONS
# -------------------------------------------------------------------------

def create_privileged_groups(
    fairness_attribs,
    value_privileged_attrib,
):
    """
    Create the privileged group definition for AIF360.
    The same privileged value is used for all protected attributes.

    Example:

        fairness_attribs = ["Sex", "Ethnicity"]
        value_privileged_attrib = 1

    produces:

        [
            {
                "Sex": 1,
                "Ethnicity": 1
            }
        ]
    """

    privileged_group = {}

    for attribute in fairness_attribs:
        privileged_group[attribute] = value_privileged_attrib

    return [privileged_group]


def create_unprivileged_groups(
    fairness_attribs,
    value_privileged_attrib,
):
    """
    Create unprivileged group definitions.

    The same privileged value is used for all protected attributes.

    For binary attributes and:

        value_privileged_attrib = 1

    creates:

        Sex = 0, Ethnicity = 1
        Sex = 1, Ethnicity = 0

    The protected attributes remain separate.
    """

    unprivileged_groups = []

    for attribute_to_change in fairness_attribs:

        group = {}

        for attribute in fairness_attribs:

            privileged_value = value_privileged_attrib

            if attribute == attribute_to_change:

                # Binary protected attributes:
                # privileged 1 -> unprivileged 0
                # privileged 0 -> unprivileged 1
                if privileged_value == 1:
                    group[attribute] = 0

                elif privileged_value == 0:
                    group[attribute] = 1

                else:
                    raise ValueError(
                        f"Protected attribute '{attribute}' must have "
                        f"a binary privileged value (0 or 1)."
                    )

            else:

                # Keep the other protected attributes at their
                # privileged value.
                group[attribute] = privileged_value

        unprivileged_groups.append(group)

    return unprivileged_groups


# -------------------------------------------------------------------------
# EQUALIZED ODDS
# -------------------------------------------------------------------------

def fit_eqodds(
    y_true,
    y_pred,
    protected_data,
    fairness_attribs,
    value_privileged_attrib,
):
    """
    Fit AIF360 Equalized Odds post-mitigation.

    The model is fitted using validation predictions from the aggregated
    ML.
    """

    dataset_true = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
    )

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
    )

    privileged_groups = create_privileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    unprivileged_groups = create_unprivileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    postprocessor = EqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        seed=42,
    )


    postprocessor = postprocessor.fit(
        dataset_true,
        dataset_pred,
    )

    return postprocessor


def apply_eqodds(
    postprocessor,
    y_true,
    y_pred,
    protected_data,
    fairness_attribs,
):
    """
    Apply a fitted Equalized Odds post-mitigation.
    """

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
    )

    dataset_mitigated = postprocessor.predict(dataset_pred)

    return dataset_mitigated.labels.flatten().astype(int)


# -------------------------------------------------------------------------
# CALIBRATED EQUALIZED ODDS
# -------------------------------------------------------------------------

def fit_calibrated_eqodds(
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
    value_privileged_attrib,
):
    """
    Fit AIF360 Calibrated Equalized Odds.

    Prediction probabilities/scores are used by this method.
    """

    dataset_true = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    # AIF360 uses the score as the prediction value for this
    # post-mitigation algorithm.
    dataset_pred.scores = np.asarray(y_score).reshape(-1, 1)

    privileged_groups = create_privileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    unprivileged_groups = create_unprivileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    postprocessor = CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        cost_constraint="weighted",
    )

    postprocessor = postprocessor.fit(
        dataset_true,
        dataset_pred,
    )

    return postprocessor


def apply_calibrated_eqodds(
    postprocessor,
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
):
    """
    Apply a fitted Calibrated Equalized Odds post-mitigation.
    """

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    dataset_pred.scores = np.asarray(y_score).reshape(-1, 1)

    dataset_mitigated = postprocessor.predict(dataset_pred)

    return dataset_mitigated.labels.flatten().astype(int)


# -------------------------------------------------------------------------
# REJECT OPTION CLASSIFICATION
# -------------------------------------------------------------------------

def fit_reject_option(
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
    value_privileged_attrib,
):
    """
    Fit AIF360 Reject Option Classification.

    This method changes predictions around the decision boundary in an
    attempt to improve fairness.
    """

    dataset_true = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    # ROC requires the prediction scores.
    dataset_pred.scores = np.asarray(y_score).reshape(-1, 1)

    privileged_groups = create_privileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    unprivileged_groups = create_unprivileged_groups(
        fairness_attribs,
        value_privileged_attrib,
    )

    postprocessor = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    postprocessor = postprocessor.fit(
        dataset_true,
        dataset_pred,
    )

    return postprocessor


def apply_reject_option(
    postprocessor,
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
):
    """
    Apply a fitted Reject Option Classification post-mitigation.
    """

    dataset_pred = create_aif360_dataset(
        y_true=y_true,
        y_pred=y_pred,
        protected_data=protected_data,
        fairness_attribs=fairness_attribs,
        scores=y_score,
    )

    dataset_pred.scores = np.asarray(y_score).reshape(-1, 1)

    dataset_mitigated = postprocessor.predict(dataset_pred)

    return dataset_mitigated.labels.flatten().astype(int)


# -------------------------------------------------------------------------
# FIT ALL REQUESTED POST-MITIGATION
# -------------------------------------------------------------------------

def fit_postmitigations(
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
    value_privileged_attrib,
    methods,
):
    """
    Fit every requested post-mitigation method independently.

    Example:

        methods = [
            "eqodds",
            "calibrated_eqodds",
            "reject_option"
        ]

    Each method is fitted using exactly the same validation data.
    """

    postmitigations = {}

    for method in methods:

        method = method.lower().strip()
        try:
            if method == "eqodds":

                postmitigations["eqodds"] = fit_eqodds(
                    y_true=y_true,
                    y_pred=y_pred,
                    protected_data=protected_data,
                    fairness_attribs=fairness_attribs,
                    value_privileged_attrib=value_privileged_attrib,
                )

            elif method == "calibrated_eqodds":

                postmitigations["calibrated_eqodds"] = (
                    fit_calibrated_eqodds(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_score=y_score,
                        protected_data=protected_data,
                        fairness_attribs=fairness_attribs,
                        value_privileged_attrib=value_privileged_attrib,
                    )
                )

            elif method == "reject_option":

                postmitigations["reject_option"] = fit_reject_option(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_score=y_score,
                    protected_data=protected_data,
                    fairness_attribs=fairness_attribs,
                    value_privileged_attrib=value_privileged_attrib,
                )

            else:

                raise ValueError(
                    f"Unknown post-mitigation method: '{method}'. "
                    f"Available methods are: "
                    f"eqodds, calibrated_eqodds, reject_option."
                )
        except Exception as e:
            print(
                f"\nWARNING: Post-mitigation method "
                f"'{method}' could not be fitted."
            )
            print("WARNING: Post-mitigation method could not be fitted.")

            for attribute in fairness_attribs:
                values = protected_data[attribute].value_counts().sort_index()

                print(f"{attribute} counts:")
                for value, count in values.items():
                    print(f"  {value}: {count}")

            print(f"Error: {e}")
            print("Continuing with the next post-mitigation method...")

            continue


    return postmitigations


# -------------------------------------------------------------------------
# APPLY ALL REQUESTED POST-MITIGATION
# -------------------------------------------------------------------------

def apply_postmitigations(
    postmitigations,
    y_true,
    y_pred,
    y_score,
    protected_data,
    fairness_attribs,
):
    """
    Apply every fitted post-mitigation independently.

    Returns
    -------
    dict

        {
            "eqodds": predictions,
            "calibrated_eqodds": predictions,
            "reject_option": predictions
        }
    """

    mitigated_predictions = {}

    for method, postmitigation in postmitigations.items():

        if method == "eqodds":

            mitigated_predictions["eqodds"] = apply_eqodds(
                postprocessor=postmitigation,
                y_true=y_true,
                y_pred=y_pred,
                protected_data=protected_data,
                fairness_attribs=fairness_attribs,
            )

        elif method == "calibrated_eqodds":

            mitigated_predictions["calibrated_eqodds"] = (
                apply_calibrated_eqodds(
                    postprocessor=postmitigation,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_score=y_score,
                    protected_data=protected_data,
                    fairness_attribs=fairness_attribs,
                )
            )

        elif method == "reject_option":

            mitigated_predictions["reject_option"] = (
                apply_reject_option(
                    postprocessor=postmitigation,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_score=y_score,
                    protected_data=protected_data,
                    fairness_attribs=fairness_attribs,
                )
            )

    return mitigated_predictions


def run_postmitigation(
    model,
    X_train,
    y_train,
    validation_idx,
    selected_features_names,
    fairness_columns_train_values,
    fairness_columns_test_values,
    y_test,
    y_pred_original,
    y_test_score,
    fairness_attribs,
    value_privileged_attrib,
    methods,
):
    """
    Fit and apply all requested post-mitigation fairness methods.

    Parameters
    ----------
    model : sklearn model
        The aggregated/global ML.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training labels.

    validation_idx : array-like
        Indices of the validation data used for fitting the
        post-mitigation.

    selected_features_names : list
        Features used by the ML.

    fairness_columns_train_values : pandas.DataFrame
        Protected attributes from the training data.

    fairness_columns_test_values : pandas.DataFrame
        Protected attributes from the test data.

    y_test : pandas.Series
        Test labels.

    y_pred_original : array-like
        Original aggregated ML predictions on the test set.

    y_test_score : array-like
        Original aggregated ML positive-class probabilities
        on the test set.

    fairness_attribs : list
        Protected attributes.

        Example:
            ["Sex", "Ethnicity"]

    value_privileged_attrib : int
    Privileged value used for the protected attributes.

    Example:
        1

    methods : list
        Post-mitigation methods to test.

        Example:
            [
                "eqodds",
                "calibrated_eqodds",
                "reject_option"
            ]

    Returns
    -------
    dict
        Mitigated predictions for every requested method.

        Example:
            {
                "eqodds": ...,
                "calibrated_eqodds": ...,
                "reject_option": ...
            }
    """

    # -------------------------------------------------------------
    # CHECK METHODS
    # -------------------------------------------------------------

    if len(methods) == 0:

        raise ValueError(
            "Post-mitigation is enabled but no methods were "
            "specified in postmitigation.methods."
        )

    # -------------------------------------------------------------
    # GET VALIDATION DATA
    # -------------------------------------------------------------

    X_val_post = X_train.iloc[
        validation_idx,
        :
    ]

    y_val_post = y_train.iloc[
        validation_idx
    ]

    protected_val = (
        fairness_columns_train_values
        .iloc[validation_idx]
        .copy()
    )

    # -------------------------------------------------------------
    # AGGREGATED ML PREDICTIONS ON VALIDATION DATA
    # -------------------------------------------------------------

    val_pred_prob = model.predict_proba(
        X_val_post[
            selected_features_names
        ]
    )

    val_score = val_pred_prob[:, 1]

    val_pred = model.predict(
        X_val_post[
            selected_features_names
        ]
    )

    # -------------------------------------------------------------
    # FIT ALL POST-MITIGATION
    # -------------------------------------------------------------

    postmitigations = fit_postmitigations(
        y_true=y_val_post,
        y_pred=val_pred,
        y_score=val_score,
        protected_data=protected_val,
        fairness_attribs=fairness_attribs,
        value_privileged_attrib=value_privileged_attrib,
        methods=methods,
    )

    # -------------------------------------------------------------
    # TEST PROTECTED ATTRIBUTES
    # -------------------------------------------------------------

    protected_test = (
        fairness_columns_test_values.copy()
    )

    # -------------------------------------------------------------
    # APPLY ALL POST-MITIGATION
    # -------------------------------------------------------------

    mitigated_predictions = apply_postmitigations(
        postmitigations=postmitigations,
        y_true=y_test,
        y_pred=y_pred_original,
        y_score=y_test_score,
        protected_data=protected_test,
        fairness_attribs=fairness_attribs,
    )

    return mitigated_predictions



def run_postmitigation_with_metrics(
    model,
    X_train,
    y_train,
    validation_idx,
    selected_features_names,
    fairness_columns_train_values,
    fairness_columns_test_values,
    y_test,
    y_pred_original,
    y_test_score,
    fairness_attribs,
    value_privileged_attrib,
    methods,
):
    """
    Run post-mitigation methods and calculate their performance
    and fairness metrics.

    If a post-mitigation method fails during fitting, its metrics
    are returned as NaN so that the expected metric keys are still
    available.
    """

    mitigated_predictions = run_postmitigation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        validation_idx=validation_idx,
        selected_features_names=selected_features_names,
        fairness_columns_train_values=fairness_columns_train_values,
        fairness_columns_test_values=fairness_columns_test_values,
        y_test=y_test,
        y_pred_original=y_pred_original,
        y_test_score=y_test_score,
        fairness_attribs=fairness_attribs,
        value_privileged_attrib=value_privileged_attrib,
        methods=methods,
    )

    metrics = {}

    # List containing all metric names produced by the
    # first successfully fitted post-mitigation method.
    metric_template = []

    # -------------------------------------------------------------
    # CALCULATE METRICS FOR SUCCESSFUL METHODS
    # -------------------------------------------------------------

    for method_name in methods:

        if method_name in mitigated_predictions:

            mitigated_pred = mitigated_predictions[method_name]

            # ---------------------------------------------------------
            # PERFORMANCE METRICS
            # ---------------------------------------------------------

            method_metrics = calculate_metrics(
                y_test,
                mitigated_pred
            )

            # ---------------------------------------------------------
            # FAIRNESS METRICS
            # ---------------------------------------------------------

            fairness_metrics = {}

            getFairnessResults(
                fairness_metrics,
                y_test.name,
                mitigated_pred,
                fairness_attribs,
                value_privileged_attrib,
                pd.concat(
                    [
                        y_test,
                        fairness_columns_test_values
                    ],
                    axis=1
                )
            )

            # Combine performance and fairness metrics
            all_method_metrics = {
                **method_metrics,
                **fairness_metrics,
            }

            # Save the complete metric structure from the first
            # successfully fitted method.
            if not metric_template:
                metric_template = list(all_method_metrics.keys())

            # Store all metrics using the expected naming convention
            for metric_name, value in all_method_metrics.items():

                metrics[
                    f"POSTMITIGATED_{method_name}_{metric_name}"
                ] = value

    # -------------------------------------------------------------
    # METHODS THAT FAILED
    # -------------------------------------------------------------

    for method_name in methods:

        if method_name not in mitigated_predictions:

            print(
                f"WARNING: Post-mitigation method '{method_name}' "
                f"was not available because it failed during fitting."
            )

            # Create all expected metric keys with NaN values
            for metric_name in metric_template:

                metrics[
                    f"POSTMITIGATED_{method_name}_{metric_name}"
                ] = np.nan

    return metrics