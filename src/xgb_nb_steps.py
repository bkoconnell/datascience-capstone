"""
Author: Brendan OConnell
Date:   May 2026

Purpose:
    For reproducibility, this module contains functions that replicate XGBoost Notebook steps.
    I was frequently running into LFS data cap issues from constantly running a NB and write a parquet.
    This module allows me to load the source parquet 1 time and then utilize functions to pass the data
    instead of always having to write it to a parquet file for processing downstream.

The XGBoost Notebooks still contain all the pertinent annotations. This module will be minimally annotated.
"""

import pandas as pd
import numpy as np
from .utils.common import sentinel_divide
from .utils.fileops import load_data_file


def xgb_feat_eng():
    """notebooks/02_feature_processing/2_engineering/feature_engineering_xgb_all_elements.ipynb"""

    # Load the NFI pre-processed dataset w/ all 89 raw elements
    nfi_df = load_data_file("preprocessed_minimal.parquet")
    meta_cols = ["stub_id", "particle_id", "class", "label", "target"]
    element_cols = [c for c in nfi_df.columns if c not in meta_cols]
    assert len(element_cols) == 89, (
        f"Expected 89 element columns, but found {len(element_cols)}."
    )

    eng_feats = pd.DataFrame()

    # Lead x Antimony feature
    eng_feats["pb_times_sb"] = nfi_df["pb"] * (nfi_df["sb"])

    # Calculate mass for denominator of ratio features
    total_mass = nfi_df[element_cols].sum(axis=1)
    # Total mass without Sb for PbBa ratio
    total_mass_no_sb = nfi_df[element_cols].sum(axis=1) - nfi_df["sb"]
    # Total mass without Ba for PbSb ratio
    total_mass_no_ba = nfi_df[element_cols].sum(axis=1) - nfi_df["ba"]
    # Total mass without Pb for BaSb ratio
    total_mass_no_pb = nfi_df[element_cols].sum(axis=1) - nfi_df["pb"]

    # Pb+Ba / (mass - Sb)
    eng_feats["pb_ba_over_non_sb_mass"] = (
        nfi_df["pb"] + nfi_df["ba"]
    ) / total_mass_no_sb

    # Pb+Sb / (mass - Ba)
    eng_feats["pb_sb_over_non_ba_mass"] = (
        nfi_df["pb"] + nfi_df["sb"]
    ) / total_mass_no_ba

    # Ba+Sb / (mass - Pb)
    eng_feats["ba_sb_over_non_pb_mass"] = (
        nfi_df["ba"] + nfi_df["sb"]
    ) / total_mass_no_pb

    # Cu+Zn / (mass)
    eng_feats["cu_zn_over_mass"] = (nfi_df["cu"] + nfi_df["zn"]) / total_mass

    # Ti+Zn / (mass)
    eng_feats["ti_zn_over_mass"] = (nfi_df["ti"] + nfi_df["zn"]) / total_mass

    # Non-Barium GSR (Pb, Sb) / Non-Barium Environmental Confounders (Ca, Si, Al, Fe)
    gsr = nfi_df["pb"] + nfi_df["sb"]
    confounders = nfi_df["ca"] + nfi_df["si"] + nfi_df["al"] + nfi_df["fe"]
    eng_feats["gsr_over_confounders"] = sentinel_divide(gsr, confounders)

    # Confirm no 'nan' or 'inf' vals
    any(
        np.isinf(eng_feats["gsr_over_confounders"])
        | eng_feats["gsr_over_confounders"].isna()
    )

    # Combine the NFI dataset with the engineered features
    ret_df = nfi_df[meta_cols + element_cols].copy()
    ret_df = pd.concat([ret_df, eng_feats], axis=1)

    return ret_df
