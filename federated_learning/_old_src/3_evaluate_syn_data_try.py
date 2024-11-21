import pandas as pd

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import warnings
from utils import evaluate_syn_data

warnings.filterwarnings("ignore")

syn_data_path_list = ['./datasets_syn/synthetic_data_type1.csv', 'datasets_syn/synthetic_data_type2.csv',
                      './datasets_syn/synthetic_data_type3.csv']

# 각 합성 데이터셋 평가
for syn_data_path in syn_data_path_list:
    df_results = evaluate_syn_data('./results/eval_results.csv',
                                   './datasets/DATOP_HF_TRANS_100_102_104_iid.csv',
                                   syn_data_path)
    print(df_results)
