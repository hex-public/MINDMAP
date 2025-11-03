import io
import json
import time
import random
from datetime import datetime
from typing import Dict, Any

import streamlit as st
from PIL import Image, ImageOps
from classifier import load_model, predict_image

@st.cache_resource
def get_model():
    return load_model("weights/best.pt")

# TITLE
st.set_page_config(
    page_title="MINDMAP",
    page_icon="🧠",
    layout="centered",
)
# INIT STATE
def init_state():
    st.session_state.setdefault("page", "info")     # info -> upload -> analysis -> result
    st.session_state.setdefault("patient_info", {})
    st.session_state.setdefault("image", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("history", [])      # 분석 로그 (관리자용)
    st.session_state.setdefault("is_admin", False)

init_state()

# --------------------- 유틸 & 스타일 ---------------------
APP_TITLE = "MINDMAP"

def app_header():
    st.markdown(
        """
        <div style="
            padding:14px 18px;
            border-radius:16px;
            background:linear-gradient(180deg,#0ea5e9, #1f2937);
            color:white;
            box-shadow:0 8px 24px rgba(0,0,0,.15);">
          <h2 style="margin:0;display:flex;align-items:center;gap:.5rem">MINDMAP</h2>
          <div style="opacity:.9;margin-top:6px;font-size:.95rem">
            MRI Brain 이미지를 기반으로 알츠하이머 예측 결과와 사용자 맞춤 약물 서비스를 제공합니다. (DEMO)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

def app_footer():
    st.write("")
    st.markdown(
        """
        <hr style="opacity:.2">
        <div style="text-align:center; font-size:.9rem; opacity:.8; position:relative;">
          2025 미래인재대학 학술제 <b>MINDMAP</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2.0rem;}
div[data-testid="stSidebar"] {border-right: 1px solid rgba(0,0,0,.07);}
div.stAlert {border-radius: 10px;}
.kbd {background:#111; color:#fff; padding:2px 6px; border-radius:6px; font-size:0.85em}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------- 비즈니스 로직 ---------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    return img

# --------------------- 사이드바 ---------------------
with st.sidebar:
    st.markdown("### Access")
    admin_name = st.text_input("Password", type="password")
    admin_toggle = st.toggle("관리자 모드", value=st.session_state.is_admin)

    if admin_toggle:
        if admin_name.strip().lower() == "admin":
            st.session_state.is_admin = True
            st.success("admin ON")
        elif admin_name != "":
            st.error("비밀번호가 틀렸습니다.")
    else:
        st.session_state.is_admin = False
        st.info("admin OFF")

    st.divider()

    if st.session_state.is_admin:
        st.markdown("### 페이지 이동")
        target = st.selectbox(
            "바로 이동",
            ["info", "upload", "analysis", "result", "admin"],
            format_func=lambda x: {
                "info": "1. 환자 정보",
                "upload": "2. MRI 업로드",
                "analysis": "3. 분석 진행",
                "result": "4. 결과",
                "admin": "관리자 대시보드",
            }[x],
        )
        if st.button("이동"):
            st.session_state.page = target
            st.rerun()

        if st.button("관리자 대시보드 열기"):
            st.session_state.page = "admin"
            st.rerun()
    else:
        st.caption("관리자 전용 기능입니다.")

# ===================== 페이지: 환자 정보 =====================
def page_info():
    app_header()
    st.title("환자 인적사항 입력")

    with st.form("patient_form", clear_on_submit=False):
        name = st.text_input("이름 *")
        age = st.number_input("나이 *", min_value=1, max_value=120, step=1)
        gender = st.radio("성별 *", ["남자", "여자"], horizontal=True)

        st.subheader("기저질환 선택")
        disease_list = ["고혈압", "당뇨", "심장질환", "간질환(간경화 등)"]
        diseases = st.multiselect("해당되는 항목을 모두 선택하세요.", disease_list)

        submitted = st.form_submit_button("Next")
    if submitted:
        master_key = name.strip().lower() == "admin"
        if not master_key and (not name or not age or not gender):
            st.warning("⚠️ 필수 항목(이름/나이/성별)을 모두 입력해주세요.")
            return

        st.session_state.patient_info = {
            "이름": name, "나이": age, "성별": gender, "기저질환": diseases,
        }
        st.session_state.page = "upload"
        st.toast("다음 단계로 이동합니다.", icon="➡️")
        st.rerun()

    app_footer()

# ===================== 페이지: 업로드 =====================
def page_upload():
    app_header()
    st.title("MRI 이미지 업로드")
    st.write("환자 정보를 바탕으로 MRI 이미지를 분석합니다.")
    st.info("[ jpg / jpeg / png ] 형식만 지원합니다.")

    uploaded_file = st.file_uploader("Image type", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="업로드된 MRI 이미지", use_container_width=True)
        if st.button("Run AI Analysis"):
            st.session_state.image = img
            st.session_state.page = "analysis"
            st.rerun()
    else:
        st.warning("⚠️ MRI 이미지를 업로드해주세요.")

    st.button("Back", on_click=lambda: st.session_state.update(page="info"))
    app_footer()

# ===================== 페이지: 분석 중 =====================
def page_analysis():
    app_header()
    st.title("🔍 AI 분석 중입니다...")

    # 진행 UI
    bar = st.progress(0, text="전처리 준비 중...")
    time.sleep(0.2)

    # 1) 전처리
    bar.progress(30, text="이미지 전처리 중...")
    img = preprocess_image(st.session_state.image)
    time.sleep(0.4)

    # 2) 모델 로딩 & 추론
    bar.progress(65, text="모델 로딩 및 추론 중...")
    model = get_model()  # @st.cache_resource 로 1회만 로드
    pred = predict_image(model, pil_image=img, imgsz=224, topk=3)
    top1 = pred["top1"]  # {"label": "...", "conf": 0.x, "index": i}

    # 3) 결과 매핑(기존 result 구조 유지)
    result = {
        "prob_alzheimer": round(top1["conf"], 4),     # 0~1 확률
        "label": top1["label"],                       # 예: "MildDemented"
        "risk": "High" if top1["conf"] >= 0.66 else ("Medium" if top1["conf"] >= 0.33 else "Low"),
        "explanations": [
            f"Top-1: {top1['label']} ({top1['conf']*100:.1f}%)",
            "모델: YOLOv8n-cls | 입력 224 | Top-K 분포는 상세에서 확인",
        ],
    }

    time.sleep(0.3)
    bar.progress(90, text="결과 취합 중...")
    time.sleep(0.2)
    bar.progress(100, text="완료!")

    # 4) 상태 저장 및 이동
    st.session_state.result = result
    st.session_state.history.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient": st.session_state.patient_info,
        "result": result,
        "topk": pred["topk"],  # 결과 페이지에서 expander로 보여주면 유용
    })
    st.session_state.page = "result"
    st.rerun()

# ===================== 페이지: 결과 =====================
def page_result():
    app_header()
    st.success("분석이 완료되었습니다.")
    res = st.session_state.result or {}

    st.subheader("예측 결과")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("예측 라벨", res.get("label", "-"))
        st.metric("위험도", res.get("risk", "-"))
    with c2:
        st.metric("알츠하이머 확률", f"{int(res.get('prob_alzheimer',0)*100)}%")

    with st.expander("환자 정보 확인"):
        st.json(st.session_state.patient_info)

    with st.expander("해설/주의"):
        for line in res.get("explanations", []):
            st.write("- " + line)

    app_footer()

# ===================== 페이지: 관리자 대시보드 =====================
def page_admin():
    app_header()
    st.title("관리자 대시보드")
    if not st.session_state.is_admin:
        st.error("접근 권한이 없습니다.")
        if st.button("돌아가기"):
            st.session_state.page = "info"
        return

    st.caption("최근 분석 로그 (세션 메모리 기반)")
    if not st.session_state.history:
        st.info("아직 로그가 없습니다.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[:20]), start=1):
            with st.expander(f"#{i} · {item['ts']} · {item['patient'].get('이름','-')}"):
                st.json(item)

    if st.button("홈으로"):
        st.session_state.page = "info"
        st.rerun()

    app_footer()

# ===================== 라우팅 =====================
PAGES = {
    "info": page_info,
    "upload": page_upload,
    "analysis": page_analysis,
    "result": page_result,
    "admin": page_admin,
}

PAGES.get(st.session_state.page, page_info)()