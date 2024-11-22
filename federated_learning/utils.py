import random

import numpy as np
# from ctgan import CTGAN
import torch
from sdv.evaluation.single_table import evaluate_quality
import os.path
import pandas as pd
from datetime import datetime
from sdmetrics.single_table import CategoricalCAP
from sdv.metadata import SingleTableMetadata
import argparse

# 시드 고정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 합성데이터 평가 함수
def evaluate_syn_data(results_path, org_data_path, syn_data_path, model_name, method, num_org, num_syn):
    columns = [
        'Timestamp', 'Model Name', 'Method', 'Num Org', 'Num Syn', 'Syn Dataset', 'Overall Quality Score',
        'Column Pair Trends',
        'BASE_YM', 'TRAN_AMT', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD',
        # 'CategoricalCAP'
    ]

    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
    else:
        df_results = pd.DataFrame(columns=columns)

    # original dataset
    df_org = pd.read_csv(org_data_path)
    # synthetic dataset
    df_syn = pd.read_csv(syn_data_path)

    # metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_org)
    metadata.update_column("BASE_YM", sdtype="datetime", datetime_format="%Y%m")
    metadata.update_column("TRAN_AMT", sdtype="numerical")
    metadata.update_column("HNDE_BANK_RPTV_CODE", sdtype="categorical")
    metadata.update_column("OPENBANK_RPTV_CODE", sdtype="categorical")
    metadata.update_column("FND_TPCD", sdtype="categorical")

    # evaluation
    quality_report = evaluate_quality(df_org, df_syn, metadata)

    validity_score = quality_report.get_score()
    column_shapes = quality_report.get_details("Column Shapes")

    print(column_shapes.columns)


    column_pair_trends = quality_report.get_details("Column Pair Trends")
    column_pair_trends_score = column_pair_trends[
        "Score"].mean() if "Score" in column_pair_trends.columns else None

    column_scores = {
        "BASE_YM": column_shapes.loc[column_shapes["Column"] == "BASE_YM", "Score"].values[0],
        "TRAN_AMT": column_shapes.loc[column_shapes["Column"] == "TRAN_AMT", "Score"].values[0],
        "HNDE_BANK_RPTV_CODE":
            column_shapes.loc[column_shapes["Column"] == "HNDE_BANK_RPTV_CODE", "Score"].values[0],
        "OPENBANK_RPTV_CODE":
            column_shapes.loc[column_shapes["Column"] == "OPENBANK_RPTV_CODE", "Score"].values[0],
        "FND_TPCD": column_shapes.loc[column_shapes["Column"] == "FND_TPCD", "Score"].values[0],
    }

    ccap_score = CategoricalCAP.compute(
         real_data=df_org,
         synthetic_data=df_syn,
         key_fields=['HNDE_BANK_RPTV_CODE'],
         sensitive_fields=['OPENBANK_RPTV_CODE', 'FND_TPCD']
    )

    new_row = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model Name': model_name,
        'Method': method,
        'Num Org': num_org,
        'Num Syn': num_syn,
        'Syn Dataset': syn_data_path,
        'Overall Quality Score': validity_score,
        'Column Pair Trends': column_pair_trends_score,
        **column_scores,
         'CategoricalCAP': ccap_score
    }
    df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    df_results.to_csv(results_path, index=False)

    return df_results

# 명령줄 인자 설정 함수
def parse_args():
    parser = argparse.ArgumentParser(description="Federated CTGAN Training Script")
    parser.add_argument("--num_samples_org", type=int, default=100, help="Number of original samples per client dataset")
    parser.add_argument("--num_samples_syn", type=int, default=300, help="Number of synthetic samples to generate")
    return parser.parse_args()
