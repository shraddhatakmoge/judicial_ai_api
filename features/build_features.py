import pandas as pd
def build_case_features(case_df, hearing_df):
    hearing_df["Adjourned"] = hearing_df["Adjourned"].astype(str).str.lower().isin(
        ["1", "true", "yes", "y"]
    ).astype(int)

    hearing_stats = hearing_df.groupby("CNR_NUMBER").agg(
        num_hearings=("HearingDate", "count"),
        avg_gap=("HearingGap_Days", "mean"),
        max_gap=("HearingGap_Days", "max"),
        adjournments=("Adjourned", "sum")
    ).reset_index()

    hearing_stats["adjournment_ratio"] = (
        hearing_stats["adjournments"] / hearing_stats["num_hearings"]
    )

    final_df = case_df.merge(
        hearing_stats, on="CNR_NUMBER", how="left"
    ).fillna(0)

    return final_df


def build_features_from_text(text: str):
    """
    TEMP feature extractor.
    Returns dummy numeric features required by the ML model.
    """
    return pd.DataFrame([{
        "num_hearings": 1,
        "avg_gap": 30,
        "max_gap": 45,
        "adjournment_ratio": 0.1
    }])
