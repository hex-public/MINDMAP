import time
import matplotlib.pyplot as plt
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps

from scripts.cam_cls import gradcam_overlay_for_cls
from classifier import load_model, predict_image, get_torch_model

@st.cache_resource
def get_model():
    return load_model("weights/best.pt")

import os
from openai import OpenAI
from streamlit.runtime.secrets import StreamlitSecretNotFoundError 
def get_openai_client():
    # 1) env
    api_key = os.getenv("OPENAI_API_KEY")

    # 2) session
    if not api_key:
        api_key = st.session_state.get("OPENAI_API_KEY")

    # 3) secrets
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (StreamlitSecretNotFoundError, KeyError):
            api_key = None

    if not api_key:
        return None

    # 진단용(앞 5자리만 표기)
    try:
        masked = api_key[:5] + "..." if len(api_key) >= 5 else "****"
        st.caption(f"OpenAI 키 감지됨: {masked}")
    except Exception:
        pass

    return OpenAI(api_key=api_key)

#=======================제목=============================
st.set_page_config(
    page_title="MINDMAP",
    page_icon="🧠",
    layout="centered",
)
#====================전역상태============================
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
    st.markdown("### 관리자")
    admin_name = st.text_input("비밀번호", type="password")
    admin_toggle = st.toggle("관리자 모드", value=st.session_state.is_admin)

    if admin_toggle:
        if admin_name.strip().lower() == "admin":
            st.session_state.is_admin = True
            st.success("관리자 모드 ON")
        elif admin_name != "":
            st.error("비밀번호가 틀렸습니다.")
    else:
        st.session_state.is_admin = False
        st.info("관리자 모드 OFF")

    st.divider()
    
    # OpenAI 키 입력 (로컬/테스트용)
    st.markdown("### OpenAI")
    _api_key_input = st.text_input("OPENAI_API_KEY", type="password")
    if _api_key_input:
        st.session_state["OPENAI_API_KEY"] = _api_key_input
        st.success("세션에 API 키 저장됨")

    if st.session_state.is_admin:
        st.markdown("### Page")

        target = st.selectbox(
            "빠른 이동",
            ["info", "upload", "analysis", "result", "report", "llm", "admin"],
            format_func=lambda x: {
                "info": "1. 환자 정보",
                "upload": "2. MRI 업로드",
                "analysis": "3. 분석 진행",
                "result": "4. 결과",
                "report": "5. 보고서",
                "llm": "6. 설명(LLM)",
                "admin": "관리자 대시보드",
            }.get(x, x),  
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
    st.title("인적사항 입력")

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
        if st.button("AI 분석하기"):
            st.session_state.image = img
            st.session_state.page = "analysis"
            st.rerun()
    else:
        st.warning("⚠️ MRI 이미지를 업로드해주세요.")

    st.button("뒤로가기", on_click=lambda: st.session_state.update(page="info"))
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

    stage = top1["label"] if top1["label"] in STAGE_DRUGS else "NonDemented"

# 3) 결과 매핑
    result = {
    "prob_alzheimer": round(top1["conf"], 4),
    "label": top1["label"],
    "risk": "High" if top1["conf"] >= 0.66 else ("Medium" if top1["conf"] >= 0.33 else "Low"),
    "stage": stage,
    "explanations": [
        f"Top-1: {top1['label']} ({top1['conf']*100:.1f}%)",
        "모델: YOLOv8n-cls | 입력 224 | Top-K 분포",
    ]

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

# --------------------- 병기별 약물 & 기저질환 규칙 ---------------------
STAGE_DRUGS = {
    "VeryMildDemented": [
        {"name": "레카네맙(Lecanemab)", "note": "아밀로이드 단백질을 줄이는 주사제입니다. 드물게 뇌 부종, 출혈이 생길 수 있어 정기 검사가 필요할 수 있습니다."},
        {"name": "세레브로리신(Cerebrolysin)", "note": "주사제로 쓰이며, 어지러움, 두통이 나타날 수 있습니다."},
        {"name": "니세르골린(Nicergoline)", "note": "어지러움이 생기거나 혈압이 내려갈 수 있습니다."},
    ],
    "MildDemented": [
        {"name": "도네페질(Donepezil)", "note": "기억, 주의에 도움을 주는 약입니다. 메스꺼움, 설사, 맥박이 느려지는 증상이 있을 수 있습니다."},
        {"name": "리바스티그민(Rivastigmine)", "note": "캡슐이나 패치로 사용합니다. 패치는 속 불편감이 비교적 적습니다."},
        {"name": "갈란타민(Galantamine)", "note": "복용 초기에 속이 불편하거나 어지러울 수 있습니다."},
    ],
    "ModerateDemented": [
        {"name": "도네페질(Donepezil)", "note": "기억, 주의에 도움을 주는 약입니다. 속 불편, 느린 맥박이 있을 수 있습니다."},
        {"name": "리바스티그민 패치(Rivastigmine Patch)", "note": "위장 부작용이 비교적 적고, 패치로 사용이 편리합니다."},
        {"name": "메만틴(Memantine)", "note": "혼란, 현기증이 드물게 생길 수 있습니다. 콩팥 기능의 검사가 필요할 수 있습니다."},
    ],
    "NonDemented": []
}


# 기저질환 규칙
def personalize_drugs(stage: str, comorbidities: list[str]) -> dict:
    base = STAGE_DRUGS.get(stage, [])
    plan = {"recommended": [], "caution": [], "avoid": []}

    for d in base:
        plan["recommended"].append((d["name"], d["note"]))

    has_htn   = "고혈압" in comorbidities
    has_dm    = "당뇨" in comorbidities
    has_heart = "심장질환" in comorbidities
    has_liver = "간질환(간경화 등)" in comorbidities

    # 고혈압
    if has_htn:
        _shift(plan, "메만틴(Memantine)", new="caution",
               reason="혈압 수치가 불균형하다면 주의가 필요합니다.")
        _shift(plan, "니세르골린(Nicergoline)", new="caution",
               reason="혈압이 내려갈 수 있습니다. 어지러움이 있을 시 복용 시간을 조절해야 합니다.")

    # 당뇨
    if has_dm:
        _annotate(plan, "리바스티그민 패치(Rivastigmine Patch)",
                  extra="속 불편이 적어 당뇨 환자도 사용할 수 있습니다.")
        _shift(plan, "갈란타민(Galantamine)", new="caution",
               reason="어지러움 증상이 지속되면 의사와 상의해야합니다.")

    # 심장질환
    if has_heart:
        for n in ["도네페질(Donepezil)", "리바스티그민(Rivastigmine)", "리바스티그민 패치(Rivastigmine Patch)", "갈란타민(Galantamine)"]:
            _shift(plan, n, new="caution",
                   reason="맥박이 느려지거나 가슴 두근거림이 생길 수 있습니다.")

    # 간질환
    if has_liver:
        _shift(plan, "도네페질(Donepezil)", new="caution",
               reason="간이 부담될 수 있어,  용량 처방에 주의가 필요합니다.")
        _shift(plan, "니세르골린(Nicergoline)", new="caution",
               reason="간 수치가 올라갈 수 있어, 정기 확인이 필요할 수 있습니다.")
        _annotate(plan, "리바스티그민 패치(Rivastigmine Patch)",
                  extra="패치제형으로, 간의 부담이 비교적 덜합니다.")
        _annotate(plan, "메만틴(Memantine)",
                  extra="주로 콩팥으로 배설돼 간질환이 있어도 대안이 될 수 있습니다.")

    if not base:
        return {"recommended": [], "caution": [], "avoid": []}

    return plan



def _shift(plan: dict, drug_name: str, new: str, reason: str):
    """recommended → caution/avoid 로 옮기고 이유 덧붙임."""
    for bucket in ("recommended", "caution", "avoid"):
        for i, (nm, note) in enumerate(plan[bucket]):
            if nm == drug_name:
                plan[bucket].pop(i)
                merged = f"{note}; {reason}" if note else reason
                plan[new].append((nm, merged))
                return

    plan[new].append((drug_name, reason))


def _annotate(plan: dict, drug_name: str, extra: str):
    """현재 버킷 유지 + 설명만 덧붙임."""
    for bucket in ("recommended", "caution", "avoid"):
        for i, (nm, note) in enumerate(plan[bucket]):
            if nm == drug_name:
                merged = f"{note}; {extra}" if note else extra
                plan[bucket][i] = (nm, merged)
                return
 
# --------------------- HTML 리포트 생성 함수 ---------------------
def build_report_html(info: dict, res: dict, plan: dict) -> str:
    # 색상
    risk = res.get("risk")
    color = "#b91c1c" if risk in ("High", "Medium") else "#166534"
    ai_text = f"<b style='color:{color}'>{res.get('label','-')}</b> · {int(res.get('prob_alzheimer',0)*100)}%"

    diseases = info.get("기저질환", []) or []
    diseases_str = ", ".join(diseases) if diseases else "없음"

    # 약물 섹션 만들기
    def _list_to_html(title, items):
        if not items:
            return f"<p><b>{title}</b>: 해당 없음</p>"
        lis = "".join([f"<li><b>{nm}</b> – {note}</li>" for nm, note in items])
        return f"<p><b>{title}</b></p><ul>{lis}</ul>"

    drugs_html = ""
    if plan and any(plan[k] for k in ("recommended", "caution", "avoid")):
        drugs_html = (
            "<h4 style='margin-top:16px'>권장 약물 & 주의사항</h4>" +
            _list_to_html("권장하는 약물", plan["recommended"]) +
            _list_to_html("주의해야 할 약물", plan["caution"]) +
            _list_to_html("피해야 할 약물", plan["avoid"])
        )
    else:
        drugs_html = "<p>본 정상군에서는 약물 치료 권장이 없습니다.</p>"

    return f"""
    <style>
      .report-box {{
        border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px;
        background: #ffffff; color:#111;
      }}
      .report-title {{ margin: 0 0 8px 0; color:#111; }}
      .report-table {{
        width: 100%; border-collapse: collapse; font-size: 15px; color:#111;
      }}
      .report-table th, .report-table td {{
        border: 1px solid #eee; padding: 10px; text-align: left; color:#111;
      }}
      .report-table th {{ width: 28%; background: #f9fafb; }}
      .report-note {{ margin-top:8px; color:#6b7280; font-size:12px; }}
    </style>
    <div class="report-box">
      <h4 class="report-title">AI 예측 결과</h4>
      <table class="report-table">
        <tr><th>환자 이름</th><td>{info.get('이름','-')}</td></tr>
        <tr><th>나이 / 성별</th><td>{info.get('나이','-')}세 / {info.get('성별','-')}</td></tr>
        <tr><th>기저질환</th><td>{diseases_str}</td></tr>
        <tr><th>YOLOv8 분석 결과</th><td>{ai_text}</td></tr>
      </table>
      {drugs_html}
      <p class="report-note">
    </div>
    """


# ===================== 페이지: 결과 =====================
def page_result():
    app_header()
    st.success("분석이 완료되었습니다.")
    res = st.session_state.result or {}
    info = st.session_state.get("patient_info", {})
    history_has_topk = bool(st.session_state.history and "topk" in st.session_state.history[-1])
    topk = st.session_state.history[-1]["topk"] if history_has_topk else None

    tab_sum, tab_topk, tab_cam = st.tabs(
        ["요약", "Top-K", "Grad-CAM"]
    )

    # 1. 요약
    with tab_sum:
        st.subheader("예측 요약")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("예측 라벨", res.get("label", "-"))
            st.metric("위험도", res.get("risk", "-"))
        with c2:
            st.metric("알츠하이머 확률", f"{int(res.get('prob_alzheimer',0)*100)}%")

        with st.expander("환자 정보"):
            st.json(info)

        with st.expander("해설/주의"):
            for line in res.get("explanations", []):
                st.write("- " + line)

    # 2. Top-K
    with tab_topk:
        st.subheader("클래스별 확률")
        if not topk:
            st.info("Top-K 결과가 없습니다.")
        else:
            labels = [x["label"] for x in topk]
            probs  = [float(x["conf"]) * 100 for x in topk]

            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.bar(labels, probs)
            ax.set_ylabel("Confidence (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate(probs):
                ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig, use_container_width=True)

    # 3. Grad-CAM
    with tab_cam:
        st.subheader("모델이 주목한 부분")
        try:
            orig_img = st.session_state.get("image")
            if orig_img is None:
                st.info("원본 이미지가 없어 Grad-CAM을 표시하지 못했습니다.")
            else:
                yolo_wrapper = get_model()
                torch_model = get_torch_model(yolo_wrapper)
                target_idx = int(topk[0]["index"]) if topk else None

                overlay_pil, _ = gradcam_overlay_for_cls(
                    torch_model=torch_model,
                    pil_image=orig_img,
                    input_size=224,
                    target_index=target_idx,
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.image(orig_img, caption="원본 이미지", use_container_width=True)
                with c2:
                    st.image(overlay_pil, caption="Grad-CAM", use_container_width=True)

        except Exception as e:
            st.warning(f"Grad-CAM 생성 중 문제가 발생했습니다.: {e}")

    # 리포트 페이지 이동
    st.divider()
    if st.button("보고서로 이동"):
        st.session_state.page = "report"
        st.rerun()

   # 하단 네비게이션
    st.write("")
    colL, colR = st.columns(2)
    with colL:
        st.button("홈으로", on_click=lambda: st.session_state.update(
            page="info", patient_info={}, image=None, result=None
        ))
    with colR:
        st.button("다시 분석하기", on_click=lambda: st.session_state.update(
            page="upload", image=None, result=None
        ))

    app_footer()

#=======================리포트======================
def page_report():
    app_header()
    st.title("보고서")

    # 결과/환자정보 없을 때
    res = st.session_state.get("result") or {}
    info = st.session_state.get("patient_info") or {}
    if not res or not info:
        st.warning("표시할 결과가 없습니다.")
        if st.button("뒤로가기"):
            st.session_state.page = "result"
            st.rerun()
        app_footer()
        return

    # 개인화 약물 플랜 생성 (result의 stage와 환자 기저질환 기반)
    stage = res.get("stage", "NonDemented")
    diseases = info.get("기저질환", []) or []
    drug_plan = personalize_drugs(stage, diseases)

    # HTML 생성 & 렌더링
    html = build_report_html(info, res, drug_plan)
    st.markdown(html, unsafe_allow_html=True)

    # HTML 다운로드
    html_bytes = html.encode("utf-8")
    st.download_button(
        "다운로드(.html)",
        data=html_bytes,
        file_name=f"mindmap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
    )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("뒤로가기"):
            st.session_state.page = "result"
            st.rerun()
    with col2:
        if st.button("홈으로"):
            st.session_state.update(page="info", patient_info={}, image=None, result=None)
            st.rerun()
    with col3:  
        if st.button("설명으로 이동"):
            st.session_state.page = "llm"
            st.rerun()


    app_footer()

# ===================== 페이지: 관리자 대시보드 =====================
def page_admin():
    app_header()
    st.title("관리자 대시보드")
    if not st.session_state.is_admin:
        st.error("접근 권한이 없습니다.")
        if st.button("뒤로가기"):
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

# ===================== LLM: ChatGPT 셋업 =====================
def build_explanation_prompt(info: dict, res: dict, plan: dict, tone: str, length: str, language: str) -> str:
    # plan dict -> 간단 나열
    def flat(bucket):
        items = plan.get(bucket, [])
        return [f"{nm} - {note}" for (nm, note) in items]

    recommended = flat("recommended")
    caution     = flat("caution")
    avoid       = flat("avoid")

    # 입력 요약
    patient = {
        "name": info.get("이름","-"),
        "age": info.get("나이","-"),
        "gender": info.get("성별","-"),
        "comorbidities": info.get("기저질환", [])
    }
    ai_result = {
        "label": res.get("label","-"),
        "risk": res.get("risk","-"),
        "prob": int(res.get("prob_alzheimer",0)*100)
    }

    tone_map = {
        "Kind": "warm, supportive, non-technical, plain language",
        "Neutral": "calm, neutral, simple wording",
        "Expertise": "professional yet patient-friendly, minimal jargon",
    }
    length_map = {
        "Short": "concise in 4-6 sentences",
        "Normal": "7-10 sentences with short paragraphs",
        "Detail": "10-15 sentences with short paragraphs and clear bullet points",
    }
    lang_tag = "Korean" if language == "한국어" else "English"

    return f"""
You are a medical explainer assistant. Output in {lang_tag}.
STYLE: {tone_map.get(tone, 'calm, neutral')}, {length_map.get(length, 'concise')}
CRITICAL RULES:
- Use ONLY the data provided below. Do NOT invent facts.
- No diagnosis or prescription. This is an educational summary for a demo.
- Prefer plain words over medical jargon. Explain terms when unavoidable.
- Structure with brief paragraphs and bullet points if helpful.
- Include a gentle disclaimer at the end.

DATA:
[Patient]
- Name: {patient['name']}
- Age: {patient['age']}
- Gender: {patient['gender']}
- Comorbidities: {', '.join(patient['comorbidities']) if patient['comorbidities'] else '없음'}

[AI Result]
- Predicted label: {ai_result['label']}
- Risk band: {ai_result['risk']}
- Estimated probability: {ai_result['prob']}%

[Medication Plan (demo rules)]
- Recommended: {recommended if recommended else ['없음']}
- Use with caution: {caution if caution else ['없음']}
- Avoid: {avoid if avoid else ['없음']}

TASK:
Write a friendly explanation that:
1) Summarizes what the AI result practically means for the user.
2) Mentions how comorbidities affect medication considerations (demo logic).
3) Highlights 2-4 key next steps users can take to talk with clinicians.
4) Avoids strong medical claims. No medication instructions or dosages.
5) Ends with a short disclaimer (e.g., '이 내용은 학술제 목적의 데모 설명입니다...').
"""
# ===================== LLM: ChatGPT 호출 =====================
def generate_llm_explanation(client, info, res, plan, tone="Kind", length="Normal", language="한국어"):
    if client is None:
        return "LLM 설정이 없어 기본 설명을 표시합니다."

    prompt = build_explanation_prompt(info, res, plan, tone, length, language)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical explainer. Keep it accurate, plain, and kind."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM 호출 실패: {e}")
        return "LLM 호출에 실패하여 기본 설명을 표시합니다."

# ===================== LLM: ChatGPT 페이지 =====================
def page_llm():
    app_header()
    st.header("LLM 설명")

    res = st.session_state.get("result") or {}
    info = st.session_state.get("patient_info") or {}
    if not res or not info:
        st.warning("분석 결과가 없습니다.")
        if st.button("뒤로가기"):
            st.session_state.page = "result"
            st.rerun()
        app_footer()
        return

    # 개인화 약물 플랜 재계산 (report와 동일 기준)
    stage = res.get("stage", "NonDemented")
    diseases = info.get("기저질환", []) or []
    plan = personalize_drugs(stage, diseases)

    col1, col2, col3 = st.columns(3)
    with col1:
        tone = st.selectbox("톤", ["친절하게", "중립적", "전문적"], index=0)
    with col2:
        length = st.selectbox("길이", ["짧게", "보통", "길게"], index=1)
    with col3:
        language = st.selectbox("언어", ["한국어", "English"], index=0)

    if st.button("LLM 설명하기"):
        with st.spinner("설명 생성 중..."):
            client = get_openai_client()
            text = generate_llm_explanation(client, info, res, plan, tone, length, language)
        st.markdown(text)

    st.write("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.button("뒤로가기", on_click=lambda: st.session_state.update(page="report"))
    with col_b2:
        st.button("홈으로", on_click=lambda: st.session_state.update(
            page="info", patient_info={}, image=None, result=None
        ))

    app_footer()


# ===================== 라우팅 =====================
PAGES = {
    "info": page_info,
    "upload": page_upload,
    "analysis": page_analysis,
    "result": page_result,
    "report": page_report,
    "llm": page_llm,
    "admin": page_admin
}

PAGES.get(st.session_state.page, page_info)()