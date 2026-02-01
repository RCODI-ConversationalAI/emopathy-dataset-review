"""
BERTopic Topic Modeling for Emotion Papers
Linux 서버에서 실행

실행 전: FINAL_EMOTION_PAPERS.csv 파일이 같은 디렉토리에 있어야 함
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BERTopic Topic Modeling - Emotion Papers")
print("=" * 60)

# 1. 데이터 로드
print("\n[1] 데이터 로드...")
df = pd.read_csv('FINAL_EMOTION_PAPERS.csv')
print(f"총 논문 수: {len(df)}")

df = df.dropna(subset=['abstract'])
df = df[df['abstract'].str.len() > 50]
df = df.reset_index(drop=True)
print(f"유효한 abstract: {len(df)}")

docs = df['abstract'].tolist()

# 2. BERTopic
print("\n[2] BERTopic 학습 중... (5-10분 소요)")
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))

topic_model = BERTopic(
    language="english",
    vectorizer_model=vectorizer,
    nr_topics=15,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)

# 3. 결과 출력
print("\n" + "=" * 60)
print("토픽 정보")
print("=" * 60)

topic_info = topic_model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']].to_string())

# 4. 저장
print("\n[3] 결과 저장...")
df['topic'] = topics
df['topic_prob'] = probs
df.to_csv('EMOTION_BERTOPIC_RESULTS.csv', index=False)
print("저장: EMOTION_BERTOPIC_RESULTS.csv")

topic_info.to_csv('EMOTION_BERTOPIC_TOPIC_INFO.csv', index=False)
print("저장: EMOTION_BERTOPIC_TOPIC_INFO.csv")

# 시각화 저장
try:
    fig = topic_model.visualize_barchart(top_n_topics=15)
    fig.write_html("topic_barchart.html")
    print("저장: topic_barchart.html")
except:
    print("시각화 저장 실패 (무시 가능)")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print("\n다음 단계:")
print("1. EMOTION_BERTOPIC_TOPIC_INFO.csv 확인")
print("2. sentiment/application 관련 토픽 번호 확인")
print("3. 해당 토픽 제외하여 최종 데이터 생성")
