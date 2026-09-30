import streamlit as st
import joblib
import pandas as pd
from openai import OpenAI

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="공정 최적화 AI 에이전트", layout="wide")
st.title("🤖 LLM 기반 공정 설정값 추천 및 불량 분석 에이전트")

# 2. 좌측 사이드바: API 키 및 파라미터 조작 UI
st.sidebar.header("🔑 보안 인증")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ 현재 장비 설정값 (Tool Input)")
layer_height = st.sidebar.slider("Layer Height (층 높이)", 0.02, 0.20, 0.10, step=0.02)
nozzle_temp = st.sidebar.slider("Nozzle Temp (노즐 온도)", 200, 250, 220, step=5)
print_speed = st.sidebar.slider("Print Speed (출력 속도)", 40, 120, 60, step=10)
fan_speed = st.sidebar.slider("Fan Speed (팬 속도)", 0, 100, 50, step=25)

# 3. 머신러닝 예측 모델 불러오기 및 실행 (Tool Use)
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

try:
    model = load_model()
    # 슬라이더 값을 바탕으로 실시간 불량률(거칠기) 예측
    input_data = pd.DataFrame({
        'layer_height': [layer_height],
        'nozzle_temperature': [nozzle_temp],
        'print_speed': [print_speed],
        'fan_speed': [fan_speed]
    })
    predicted_roughness = model.predict(input_data)[0]
    
    st.metric(label="📊 AI가 예측한 표면 거칠기 (높을수록 불량)", value=f"{predicted_roughness:.2f}")
except Exception as e:
    st.error("머신러닝 모델을 불러오는 데 실패했습니다. 먼저 train_model.py를 실행하세요.")

st.markdown("---")

# 4. LLM 에이전트 채팅 UI (Reasoning & Feedback)
st.subheader("💬 AI 에이전트에게 원인 분석 요청하기")

# 채팅 기록을 저장하는 메모리 (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 현재 장비 설정값을 확인했습니다. 생산 중 발생한 불량 증상을 말씀해주시면, 데이터를 분석하여 최적의 파라미터를 역산해 드립니다."}
    ]

# 이전 채팅 기록 화면에 띄우기
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자가 채팅창에 질문을 입력했을 때 작동하는 로직
user_input = st.chat_input("증상 입력 (예: 출력물 표면이 너무 거칠게 나옵니다. 어떻게 수정할까요?)")

if user_input:
    if not api_key:
        st.error("좌측 사이드바에 OpenAI API Key를 먼저 입력해주세요!")
    else:
        # 사용자 질문 화면에 출력
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # 에이전트에게 상황 판단(Reasoning)을 맡기는 프롬프트 엔지니어링
        system_prompt = f"""
        당신은 스마트팩토리의 3D 프린터 및 공정 최적화 전문가 AI 에이전트입니다.
        [현재 작업 환경 데이터]
        - 층 높이: {layer_height}
        - 노즐 온도: {nozzle_temp}
        - 출력 속도: {print_speed}
        - 쿨링팬 속도: {fan_speed}
        - 내부 머신러닝이 예측한 표면 거칠기 점수: {predicted_roughness:.2f}

        [행동 지침]
        1. 사용자의 불만(입력)을 파악하세요.
        2. '현재 작업 환경 데이터'와 '예측된 거칠기 점수'를 공학적 근거로 삼아 원인을 추론하세요.
        3. 해결을 위해 온도를 몇 도 올려야 할지, 속도를 얼마나 줄여야 할지 직관적이고 구체적인 수치로 지시하세요.
        """

        # OpenAI API 통신
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 가볍고 빠른 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        
        # AI 답변 화면에 출력
        ai_answer = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        st.chat_message("assistant").write(ai_answer)