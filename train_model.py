import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("데이터 불러오는 중...")
# 파일명이 data.csv.csv 이므로 그대로 불러옵니다.
df = pd.read_csv('data.csv')

# 빈 열(,,,) 무시하고 딱 필요한 5개 컬럼만 선택
columns_to_use = ['layer_height', 'nozzle_temperature', 'print_speed', 'fan_speed', 'roughness']
df = df[columns_to_use]

# 혹시 모를 빈칸(결측치) 제거하여 에러 방지
df = df.dropna()

# X(입력 파라미터 4개)와 y(예측할 불량 지표 1개) 분리
X = df[['layer_height', 'nozzle_temperature', 'print_speed', 'fan_speed']]
y = df['roughness']

print("XGBoost 대체용 랜덤 포레스트 AI 모델 학습 중...")
# 머신러닝 회귀 모델 학습 (CPU로 1초면 충분함)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 학습된 모델을 pkl 파일로 저장 (LLM 에이전트가 함수로 호출할 파일)
joblib.dump(model, 'rf_model.pkl')
print("✅ 모델 학습 완료! 'rf_model.pkl' 파일이 생성되었습니다.")