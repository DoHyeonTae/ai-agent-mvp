# 🤖 3D 프린터 공정 최적화 AI 에이전트

## 📌 프로젝트 기획 의도
스마트팩토리 환경에서 3D 프린팅 공정 중 발생하는 **표면 거칠기(Roughness) 불량 문제**를 해결하기 위한 AI 에이전트 MVP입니다. 기존에는 작업자의 직관과 감에 의존해 장비 설정값을 맞췄으나, 본 프로젝트는 머신러닝 예측과 LLM의 추론 능력을 결합하여 **데이터 기반의 최적 파라미터를 추천**하고 공정 불량률을 최소화하는 것을 목표로 합니다.

## 💡 핵심 기능
1. **실시간 불량 예측 (Tool Use)**
   * `Layer Height`, `Nozzle Temp`, `Print Speed`, `Fan Speed` 4가지 핵심 장비 파라미터를 입력하면, 학습된 Random Forest 모델이 표면 거칠기(불량 지표)를 실시간으로 예측합니다.
2. **LLM 기반 원인 분석 및 피드백 (Reasoning & Feedback)**
   * 예측된 수치와 현재 파라미터 상태를 분석하여, 불량 증상에 대해 "노즐 온도를 X도 올리고, 속도를 Y만큼 줄이세요"와 같은 구체적이고 공학적인 해결책을 채팅 형태로 제시합니다.

## 🛠️ 적용된 기술 스택
* **Language & UI:** Python, Streamlit
* **Machine Learning:** Scikit-learn (RandomForestRegressor), Pandas
* **AI Agent (LLM):** OpenAI API (GPT-4o-mini)
* **Deployment:** Streamlit Community Cloud

## 🚀 배포 링크
* [여기서 AI 에이전트 웹 앱을 직접 체험해 보세요!](https://3d-printer-optimizer-mvp.streamlit.app/)
