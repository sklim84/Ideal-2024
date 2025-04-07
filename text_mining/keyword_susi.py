from keybert import KeyBERT
from utils import filter_text_columns
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def create_wordcloud(df_data, cond_col_name, cond_col_codes, target_col_name, wc_results_path):
    # Filtering
    df_filtered = df_data[df_data[cond_col_name].isin(cond_col_codes)]

    documents = df_filtered[target_col_name].fillna('').tolist()
    documents = [str(doc) for doc in documents if doc.strip()]
    documents = ' '.join(documents)
    # 특수문자 등 제거
    text = re.sub('[^A-Za-z0-9가-힣_ ]+', '', documents)
    # print(f'##### text: {text}')

    font_path = '/home/bigdyl/anaconda3/envs/tm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/NanumGothic.ttf'
    wc = WordCloud(background_color='white', font_path=font_path, max_words=1000).generate(text)
    # wc = WordCloud(background_color='white', max_words=1000).generate(text)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(wc_results_path)
    plt.show()


def run_keyword_analysis(df_data, cond_col_name, cond_col_codes, target_col_name, top_n, results_path):
    # Filtering
    df_filtered = df_data[df_data[cond_col_name].isin(cond_col_codes)]

    documents = df_filtered[target_col_name].fillna('').tolist()
    documents = [str(doc) for doc in documents if doc.strip()]
    documents = ' '.join(documents)
    # 특수문자 등 제거
    documents = re.sub('[^A-Za-z0-9가-힣_ ]+', '', documents)
    # print(documents)

    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(documents, top_n=top_n)
    print(keywords)

    # Save keywords
    df_keywords = pd.DataFrame(keywords, columns=['keyword', 'score'])
    df_keywords.to_csv(results_path, index=False, encoding='utf-8-sig')  # Save to CSV


if __name__ == "__main__":

    data_name = 'BIZ_CON_SUSI_RES'
    data_file = 'BIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx'
    cond_col_name = '수행사항종합결과'
    cond_col_codes = [1]
    text_density_threshold = 0.5
    unique_text_threshold = 20

    data_path = f'./datasets/{data_file}'
    df_data = pd.read_excel(data_path, index_col=None)

    top_n = 20

    for group_code in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # 그룹별 발췌
        df_data_group = df_data[df_data['그룹구분'] == group_code]
        print(f'group {group_code} filtering: {len(df_data)} → {len(df_data_group)}')

        # 텍스트 밀도 기반 컬럼 필터링
        df_data_group_fil = filter_text_columns(df_data_group, excep_col_names=[cond_col_name],
                                          text_density_threshold=text_density_threshold,
                                          unique_text_threshold=unique_text_threshold)
        print(f'column filtering: {len(df_data_group.columns)} → {len(df_data_group_fil.columns)}')

        target_col_names = [col for col in df_data_group_fil.columns if col != cond_col_name]

        for target_col_name in target_col_names:
            print(f'Keyword analysis start: {target_col_name}')
            results_path = f'./results/keyword/{data_name}/group_{group_code}_{target_col_name}_keywords.csv'
            run_keyword_analysis(df_data_group_fil, cond_col_name, cond_col_codes, target_col_name, top_n, results_path)

            wc_results_path = f'./results/keyword/{data_name}/wordcloud/group_{group_code}_{target_col_name}_wordcloud.png'
            create_wordcloud(df_data_group_fil, cond_col_name, cond_col_codes, target_col_name, wc_results_path)
