import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from wordcloud import WordCloud

from utils import filter_text_columns


def create_wordcloud(model, topic, wc_results_path):
    # print(matplotlib.__file__)

    text = {word: value for word, value in model.get_topic(topic)}

    font_path = '/home/bigdyl/anaconda3/envs/tm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/NanumGothic.ttf'
    wc = WordCloud(background_color='white', font_path=font_path, max_words=1000).generate_from_frequencies(text)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(wc_results_path)
    plt.show()


def run_topic_modeling(df_data, cond_col_name, cond_col_codes, target_col_name, results_path):
    # Filtering
    df_filtered = df_data[df_data[cond_col_name].isin(cond_col_codes)]

    documents = df_filtered[target_col_name].values.tolist()
    documents = [str(doc) for doc in documents if doc is not None and doc is not np.nan]

    # BERTopic
    topic_model = BERTopic(language='multilingual')
    topics, probs = topic_model.fit_transform(documents)

    # Save results
    topic_keywords = []
    unique_topics = sorted(set(topics))
    for topic in unique_topics:
        keywords = topic_model.get_topic(topic)
        if keywords:
            keywords_str = ' '.join([word for word, _ in keywords])
            topic_keywords.append((topic, keywords_str))
    df_topics = pd.DataFrame(topic_keywords, columns=['topic number', 'keywords'])
    print(df_topics)
    df_topics.to_csv(results_path, index=False)

    return topic_model, unique_topics


if __name__ == "__main__":

    data_name = 'BIZ_CON_RES'
    data_file = 'BIZ_CONSULTING_RESULT_202411101625.xlsx'
    cond_col_name = '종합의견'
    cond_col_codes = [3]
    text_density_threshold = 0.5
    unique_text_threshold = 100

    data_path = f'./datasets/{data_file}'
    df_data = pd.read_excel(data_path, index_col=None)

    # 텍스트 밀도 기반 컬럼 필터링
    df_data_fil = filter_text_columns(df_data, excep_col_names=[cond_col_name],
                                      text_density_threshold=text_density_threshold,
                                      unique_text_threshold=unique_text_threshold)
    print(f'##### column filtering: {len(df_data.columns)} → {len(df_data_fil.columns)}')

    target_col_names = [col for col in df_data_fil.columns if col != cond_col_name]
    for target_col_name in target_col_names:
        print(f'topic modeling start: {target_col_name}')

        # 토픽 모델링
        topic_results_path = f'./results/topic/{data_name}/{target_col_name}_topic.csv'
        topic_model, topics = run_topic_modeling(df_data_fil, cond_col_name, cond_col_codes, target_col_name,
                                                 topic_results_path)

        # 워드 클라우드
        for topic in topics:
            wc_results_path = f'./results/topic/{data_name}/wordcloud/{target_col_name}_topic_{topic}.png'
            create_wordcloud(topic_model, topic, wc_results_path)
