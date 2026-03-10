from io import StringIO

import pandas as pd


def get_conversion_df():
    data = (
        "Assay,Log_Scale,Multiplier,Log_name\n"
        "LogD,False,1,LogD\n"
        "KSOL,True,1e-6,LogS\n"
        "HLM CLint,True,1,Log_HLM_CLint\n"
        "MLM CLint,True,1,Log_MLM_CLint\n"
        "Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB\n"
        "Caco-2 Permeability Efflux,True,1,Log_Caco_ER\n"
        "MPPB,True,1,Log_Mouse_PPB\n"
        "MBPB,True,1,Log_Mouse_BPB\n"
        "MGMB,True,1,Log_Mouse_MPB\n"
    )
    s = StringIO(data)
    conversion_df = pd.read_csv(s)
    return conversion_df


def inverse_log_transform_assay_data(test_df):
    """
    Reverse the log-transformation of assay columns using the predefined mapping.

    Args:
        test_df (pd.DataFrame): Input DataFrame with log-transformed assay columns.
                                Must have columns ["SMILES", "Molecule Name", ...log assay columns...]

    Returns:
        pd.DataFrame: A DataFrame with the original assay values recovered.
    """
    conversion_df = get_conversion_df()
    reverse_dict = dict(
        [(x[-1], x[0:-1]) for x in conversion_df.values]
    )

    output_df = test_df[["SMILES", "Molecule Name"]].copy()

    for col in test_df.columns[2:]:
        if col == "dataset":
            continue
        if col not in reverse_dict:
            print(f"Skipping unrecognized column: {col}")
            continue

        orig_name, log_scale, multiplier = reverse_dict[col]
        log_scale = str(log_scale).lower() == "true"
        multiplier = float(multiplier)

        output_df[orig_name] = test_df[col].astype(float)
        if log_scale:
            output_df[orig_name] = (
                (10 ** output_df[orig_name]) * (1 / multiplier) - 1
            ).clip(lower=0)

    return output_df, reverse_dict
