import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import streamlit.components.v1 as components

import io
import numpy as np

from PIL import Image, ImageOps

from scripts.cam_cls import gradcam_overlay_for_cls
from classifier import load_model, predict_image, get_torch_model

@st.cache_resource
def get_model():
    return load_model("weights/best.pt")

import os
from openai import OpenAI

# 맨 위 한 번만(로컬에서 import 에러 방지용)
try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:
    class StreamlitSecretNotFoundError(Exception):
        pass

def get_openai_client():
    import os
    from openai import OpenAI

    # 1) ENV
    api_key = os.getenv("OPENAI_API_KEY")

    # 2) SESSION
    if not api_key:
        api_key = st.session_state.get("OPENAI_API_KEY")

    # 3) SECRETS (파일이 없으면 예외 나므로 try로 보호)
    if not api_key:
        try:
            # "in" 체크도 파싱을 유발할 수 있어 접근 전체를 try로 감싼다
            api_key = st.secrets["OPENAI_API_KEY"]  # 키 없으면 KeyError, 파일 없으면 StreamlitSecretNotFoundError
        except (StreamlitSecretNotFoundError, KeyError):
            api_key = None

    if not api_key:
        return None

    # (선택) 마스킹 표시
    try:
        masked = api_key[:5] + "..." if len(api_key) >= 8 else "****"
        st.sidebar.write(f"OpenAI 키 감지됨: {masked}")
    except Exception:
        pass

    return OpenAI(api_key=api_key)

# ====== 단계/위험도 표시 유틸 ======
STAGE_KO_MAP = {
    "VeryMildDemented": "매우 경도(초기) 치매 의심",
    "MildDemented": "경도 치매 의심",
    "ModerateDemented": "중등도 치매 의심",
    "NonDemented": "치매 비의심(정상 범주)"
}

def stage_to_korean(label: str) -> str:
    return STAGE_KO_MAP.get(label, label or "-")

def risk_text(prob_float: float) -> str:
    try:
        pct = int(float(prob_float) * 100)
    except Exception:
        pct = 0
    if pct >= 66:
        bucket = "High"
    elif pct >= 33:
        bucket = "Medium"
    else:
        bucket = "Low"
    return f"{pct}% ({bucket})"

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

    # 성별 먼저 선택 → 변경 시 즉시 rerun
    gender = st.radio("성별 *", ["남자", "여자"], horizontal=True, key="pf_gender")

    with st.form("patient_form", clear_on_submit=False):
        name = st.text_input("이름 *", key="pf_name")
        age = st.number_input("나이 *", min_value=1, max_value=120, step=1, key="pf_age")

        st.subheader("건강 상태 / 기저 질환")
        disease_list = ["고혈압", "당뇨", "심장질환", "간질환(간경화 등)"]
        if st.session_state.get("pf_gender") == "여자":
            disease_list.append("임신(임산부)")
        diseases = st.multiselect("해당되는 항목을 모두 선택하세요.", disease_list, key="pf_diseases")

        submitted = st.form_submit_button("Next")

    if submitted:
        master_key = (st.session_state.get("pf_name","").strip().lower() == "admin")
        if not master_key and (not st.session_state.get("pf_name") or not st.session_state.get("pf_age") or not st.session_state.get("pf_gender")):
            st.warning("⚠️ 필수 항목(이름/나이/성별)을 모두 입력해주세요.")
            return

        st.session_state.patient_info = {
            "이름": st.session_state.get("pf_name"),
            "나이": st.session_state.get("pf_age"),
            "성별": st.session_state.get("pf_gender"),
            "기저질환": st.session_state.get("pf_diseases", []),
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

    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.read()
            if not img_bytes:
                raise ValueError("업로드된 파일이 비어 있습니다.")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(img, caption="업로드된 MRI 이미지", use_container_width=True)

            if st.button("AI 분석하기"):
                st.session_state.image_bytes = img_bytes
                st.session_state.image = img
                st.session_state.page = "analysis"
                st.rerun()
        except Exception as e:
            st.error(f"이미지 처리 중 오류: {e}")
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

# 한 줄 공통 메시지(짧고 부드럽게)
PREG_COMMON = "임신·수유 시에는 약 사용을 조금 더 신중히 결정해요. 필요하면 의료진과 꼭 상의해주세요."

# 약물별 짧은 코멘트(친절 톤)
PREGNANCY_NOTES = {
    "레카네맙(Lecanemab)": "임신 중 자료가 충분하지 않습니다.",
    "세레브로리신(Cerebrolysin)": "임부 대상 자료가 아직 많지 않습니다.",
    "갈란타민(Galantamine)": "임부 임상자료는 부족하나, 동물시험상 큰 이상 보고는 적습니다."
}

# 기저질환 규칙
def personalize_drugs(stage: str, comorbidities: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    base = STAGE_DRUGS.get(stage, [])
    plan = {"recommended": [], "caution": [], "avoid": []}

    # 기본 추천 적재
    for d in base:
        plan["recommended"].append((d["name"], d["note"]))

    # 표기 차이 허용: "임신" 또는 "임신(임산부)"
    has_preg  = any(x in comorbidities for x in ("임신", "임신(임산부)"))
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
               reason="어지러움 증상이 지속되면 의사와 상의해야 합니다.")

    # 심장질환
    if has_heart:
        for n in ["도네페질(Donepezil)", "리바스티그민(Rivastigmine)",
                  "리바스티그민 패치(Rivastigmine Patch)", "갈란타민(Galantamine)"]:
            _shift(plan, n, new="caution",
                   reason="맥박이 느려지거나 가슴 두근거림이 생길 수 있습니다.")

    # 간질환
    if has_liver:
        _shift(plan, "도네페질(Donepezil)", new="caution",
               reason="간이 부담될 수 있어 용량 처방에 주의가 필요합니다.")
        _shift(plan, "니세르골린(Nicergoline)", new="caution",
               reason="간 수치가 올라갈 수 있어 정기 확인이 필요할 수 있습니다.")
        _annotate(plan, "리바스티그민 패치(Rivastigmine Patch)",
                  extra="패치 제형으로 간 부담이 비교적 덜합니다.")
        _annotate(plan, "메만틴(Memantine)",
                  extra="주로 콩팥으로 배설돼 간질환이 있어도 대안이 될 수 있습니다.")

    # 임신: recommended → caution 전환 + 공통 경고 + 약물별 주의 메모
    if has_preg:
        for nm, note in list(plan["recommended"]):
            extra = "임신/수유 가능성이 있으면 반드시 전문의와 상의하세요."
            if nm in PREGNANCY_NOTES:
                extra = f"{extra} {PREGNANCY_NOTES[nm]}"
            _shift(plan, nm, new="caution", reason=extra)

        # 이미 caution에 있던 항목에도 주의 문구 보강
        for i, (nm, note) in enumerate(list(plan["caution"])):
            extra = "임신/수유 가능성이 있으면 반드시 전문의와 상의하세요."
            add = PREGNANCY_NOTES.get(nm)
            if add and add not in note:
                note = f"{note}; {extra} {add}"
            elif extra not in note:
                note = f"{note}; {extra}"
            plan["caution"][i] = (nm, note)

    # NonDemented 등
    if not base:
        return {"recommended": [], "caution": [], "avoid": []}

    # 중복 제거(규칙 다중 적용 대비)
    _dedup_plan(plan)
    return plan

def _dedup_plan(plan: dict):
    for bucket in ("recommended", "caution", "avoid"):
        seen = {}
        for nm, note in plan[bucket]:
            if nm in seen:
                if note and note not in seen[nm]:
                    seen[nm] = f"{seen[nm]}; {note}"
            else:
                seen[nm] = note
        plan[bucket] = [(k, v) for k, v in seen.items()]


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

# --------------------- HTML 리포트 생성 함수 ---------------------
from typing import List, Tuple  # 상단에 이미 있으면 중복 import 불필요

def build_report_html(info: dict, res: dict, plan: dict) -> str:
    """보고서 HTML을 생성해 반환한다. (임신 시 약물 전부 '주의' 처리 + 안내 배너 표시)"""
    risk = res.get("risk")
    color = "#b91c1c" if risk in ("High", "Medium") else "#166534"
    ai_text = (
        f"<span style='font-weight:bold; color:{color};'>"
        f"{res.get('label','-')}</span> · {int(res.get('prob_alzheimer',0)*100)}%"
    )

    diseases = info.get("기저질환", []) or []
    diseases_str = ", ".join(diseases) if diseases else "없음"
    has_preg = any(x in diseases for x in ("임신", "임신(임산부)"))

    # --- 임신/수유일 때: 모든 권장 → 주의로 강제 이동(서버 로직 중복 대비, 여기서도 보정) ---
    if has_preg and plan:
        rec_list = list(plan.get("recommended", []))
        if rec_list:
            plan["recommended"] = []  # 권장 비우고
            # 기존 주의 리스트에 임신 주의 문구를 덧붙여 이동
            caution = plan.get("caution", [])
            for nm, note in rec_list:
                extra = "임신/수유 가능성이 있으면 반드시 전문의와 상의하세요."
                if "임신/수유" not in (note or ""):
                    note = f"{(note or '').strip()}{'; ' if note else ''}{extra}"
                caution.append((nm, note))
            plan["caution"] = caution

        # 이미 주의에 있는 항목에도 안내문이 없다면 보강
        if "caution" in plan:
            new_caution = []
            for nm, note in plan["caution"]:
                if "임신/수유" not in (note or ""):
                    note = f"{(note or '').strip()}{'; ' if note else ''}임신/수유 가능성이 있으면 반드시 전문의와 상의하세요."
                new_caution.append((nm, note))
            plan["caution"] = new_caution

    # --- 약물 카드 HTML 생성 (비어있으면 섹션 숨김) ---
    from typing import List, Tuple
    def _cards_html(bucket_title: str, items: List[Tuple[str, str]], badge_class: str) -> str:
        if not items:
            return ""  # 비어있으면 아예 표시 안 함
        cards = []
        for drug, note in items:
            cards.append(
                f"""
                <div class="drug-card">
                  <div class="drug-badge {badge_class}">{bucket_title}</div>
                  <div class="drug-name">{drug}</div>
                  <div class="drug-note">{note}</div>
                </div>
                """
            )
        return "".join(cards)

    has_any = bool(plan) and any(plan.get(k) for k in ("recommended", "caution", "avoid"))
    if has_any:
        rec_cards = _cards_html("권장", plan.get("recommended", []), "rec")
        cau_cards = _cards_html("주의",  plan.get("caution", []),     "cau")
        avd_cards = _cards_html("피함",  plan.get("avoid", []),        "avd")

        # 상단 카운트
        n_rec = len(plan.get("recommended", []))
        n_cau = len(plan.get("caution", []))
        n_avd = len(plan.get("avoid", []))

        # 임신 안내 배너(있을 때만)
        preg_banner = ""
        if has_preg:
            preg_banner = """
            <div class="alert-preg">
              임신/수유 관련 안내: 임신/수유 시 대부분의 약물은 신중히 사용해야 하므로,
              아래 약물은 ‘주의’ 항목으로 표시됩니다.
            </div>
            """

        # 전부 비어있으면 섹션 숨김
        if not (rec_cards or cau_cards or avd_cards):
            drugs_html = preg_banner
        else:
            drugs_html = f"""
            {preg_banner}
            <div class="drug-section">
              <h4 class="drug-title">💊 약물 요약
                <span class="chip rec">권장 {n_rec}</span>
                <span class="chip cau">주의 {n_cau}</span>
                <span class="chip avd">피함 {n_avd}</span>
              </h4>
              <div class="drug-grid">
                {rec_cards}{cau_cards}{avd_cards}
              </div>
            </div>
            """
    else:
        drugs_html = ""  # NonDemented 등: 섹션 자체 숨김

    return f"""
    <style>
      .report-box {{
        border: 2px solid #333;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        background-color: #ffffff;
      }}
      .report-header {{
        text-align: center;
        border-bottom: 2px solid #ddd;
        padding-bottom: 10px;
        margin-bottom: 15px;
      }}
      .report-header h3 {{ margin: 0; color: #1E90FF; }}
      .report-header p {{ font-size: 12px; color: #555; }}

      .report-table {{
        width: 100%; border-collapse: collapse; margin-top: 10px;
      }}
      .report-table th, .report-table td {{
        border: 1px solid #eee; padding: 10px; text-align: left; font-size: 15px; color: #111;
      }}
      .report-table th {{
        background-color: #f8f8f8; width: 30%; font-weight: bold; color: #333;
      }}
      .important-result td {{ background-color: #fffacd; font-size: 16px; }}

      /* --- 안내 배너(임신/수유) --- */
      .alert-preg {{
        margin-top: 14px;
        padding: 10px 12px;
        border-radius: 8px;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #ffedd5;
        font-size: 13px;
      }}

      /* --- 약물 섹션 스타일 --- */
      .drug-section {{ margin-top: 16px; }}
      .drug-title {{ margin: 0 0 10px 0; display:flex; align-items:center; gap:8px; }}
      .chip {{
        display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600;
        border:1px solid rgba(0,0,0,.08);
      }}
      .chip.rec {{ background:#ecfdf5; color:#065f46; border-color:#d1fae5; }}
      .chip.cau {{ background:#fff7ed; color:#9a3412; border-color:#ffedd5; }}
      .chip.avd {{ background:#fef2f2; color:#991b1b; border-color:#fee2e2; }}

      .drug-grid {{
        display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
        gap:12px; margin-top:6px;
      }}
      .drug-card {{
        border:1px solid #e5e7eb; border-radius:10px; padding:12px;
        background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.03);
      }}
      .drug-badge {{
        display:inline-block; font-size:11px; font-weight:700; letter-spacing:.2px;
        padding:2px 6px; border-radius:6px; margin-bottom:6px;
      }}
      .drug-badge.rec {{ background:#ecfdf5; color:#065f46; }}
      .drug-badge.cau {{ background:#fff7ed; color:#9a3412; }}
      .drug-badge.avd {{ background:#fef2f2; color:#991b1b; }}

      .drug-name {{ font-weight:700; margin-bottom:4px; }}
      .drug-note {{ font-size:13px; color:#444; line-height:1.45; }}

      .report-note {{ margin-top: 12px; color: #6b7280; font-size: 12px; }}
    </style>

    <div class="report-box">
      <div class="report-header">
        <h3>MINDMAP</h3>
        <p>알츠하이머 AI 예측 결과</p>
      </div>

      <table class="report-table">
        <tr><th>환자 이름</th><td>{info.get('이름','-')}</td></tr>
        <tr><th>나이 / 성별</th><td>{info.get('나이','-')}세 / {info.get('성별','-')}</td></tr>
        <tr><th>건강 상태 / 기저질환</th><td>{diseases_str}</td></tr>
        <tr class="important-result"><th>YOLOv8 분석 결과</th><td>{ai_text}</td></tr>
      </table>

      {drugs_html}

      <p class="report-note">※ 본 결과는 AI 분석 결과이며, 최종적인 판단은 전문의 상담이 필요합니다.</p>
    </div>
    """


# ======================= 페이지: 보고서 =======================
def page_report():
    app_header()
    st.title("보고서")

    res  = st.session_state.get("result") or {}
    info = st.session_state.get("patient_info") or {}

    if not res or not info:
        st.warning("표시할 결과가 없습니다.")
        if st.button("뒤로가기"):
            st.session_state.page = "result"
            st.rerun()
        app_footer()
        return

    # 최신 개인화 플랜 계산
    stage    = res.get("stage", "NonDemented")
    diseases = info.get("기저질환", []) or []
    plan     = personalize_drugs(stage, diseases)

    # 1) HTML 생성
    report_html = build_report_html(info, res, plan)

    # 2) HTML 렌더 (아이프레임 격리: CSS 충돌 방지)
    components.html(
        html=report_html,
        height=900,         # 길면 1000~1200으로 조절
        scrolling=True,     # 내부 스크롤 허용
    )

    # 3) 다운로드 버튼
    st.download_button(
        label="HTML 보고서 다운로드",
        data=report_html.encode("utf-8"),
        file_name=f"{info.get('이름','환자')}_AI_치매_예측_보고서.html",
        mime="text/html",
    )

    # 네비게이션
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("결과로 돌아가기"):
            st.session_state.page = "result"
            st.rerun()
    with col2:
        if st.button("홈으로"):
            st.session_state.update(page="info", patient_info={}, image=None, result=None)
            st.rerun()
    with col3:
        if st.button("LLM 설명으로 이동"):
            st.session_state.page = "llm"
            st.rerun()

    app_footer()

#=======================ChatGPT====================
def build_explanation_prompt(info: dict, res: dict, plan: dict, tone: str, length: str, language: str):
    """설명문에서 '환자/사용자' 금지, 오직 이름(…님) 호칭만 사용."""
    name = (info.get("이름") or "이름 미기입").strip()
    age  = info.get("나이","-")
    gender = info.get("성별","-")
    comorbidities = info.get("기저질환", []) or []

    label = res.get("label","-")
    prob  = res.get("prob_alzheimer", 0.0)

    stage_ko  = stage_to_korean(label)
    risk_line = risk_text(prob)  # 예: "78% (High)"

    def flat(bucket: str):
        items = plan.get(bucket, [])
        return [f"{nm} - {note}" for (nm, note) in items] or ["없음"]

    recommended = flat("recommended")
    caution     = flat("caution")
    avoid       = flat("avoid")

    # 톤/형식 가이드: 반드시 "{name}님"으로 호칭, '환자/사용자' 금지
    tone_guides = {
        "Kind":      f"{name}님에게 직접 말하듯 따뜻하고 공감 있게 서술하고, 전문용어는 쉬운 말로 풀어쓴다.",
        "Neutral":   f"객관적이고 간결하게 사실을 전달하되, 문장은 자연스럽고 읽기 쉽게 쓴다.",
        "Expertise": f"전문가 보고서 톤으로 병태생리·약물기전·부작용·모니터링을 구체적으로 적되 가독성을 유지한다.",
    }
    length_guides = {
        "Short":  "4~6문장 한 단락. 줄바꿈/불릿 금지. 첫 문장에 단계와 위험도 %를 요약.",
        "Normal": "1~2단락. 필요 시 짧은 불릿(최대 3개) 허용. 첫 문단 첫 문장에 단계와 위험도 %를 요약.",
        "Detail": "## 소제목 섹션(요약, 임상 해석, 약물 선택, 주의, 생활 조언). 각 섹션 2~5문장 또는 짧은 불릿. 요약 섹션 첫 줄에 단계와 위험도 % 명시.",
    }

    # 임신/수유 블록
    preg_block = ""
    if any(x in comorbidities for x in ("임신", "임신(임산부)")):
        preg_block = (
            "- 임신/수유 가능성이 있으면 약물 사용 전 전문의 상담 필요. "
            "레카네맙: 임부 자료 부족 / 세레브로리신: 임부 자료 부족 / 갈란타민: 임상자료 제한적."
        )

    # 컨텍스트(LLM에게만 보여줄 데이터)
    context = f"""
[참고데이터 — 그대로 복사하지 말 것]
- 이름: {name} / {age}세 / {gender} / 기저질환: {', '.join(comorbidities) if comorbidities else '없음'}
- AI 예측: 단계={label} → {stage_ko}, 위험도={risk_line}
- 약물 권장: {recommended}
- 약물 주의: {caution}
- 약물 피함: {avoid}
- 임신 주의: {('해당' if preg_block else '해당 없음')}
"""

    # 출력 규칙
    hard_rules = f"""
[출력 규칙 — 매우 중요]
1) 모든 출력은 {language}로 작성.
2) 호칭은 반드시 '{name}님'만 사용하고, '환자'나 '사용자'라는 단어는 절대 쓰지 말 것.
3) 첫 문장(또는 요약 섹션 1행)에 단계('{stage_ko}')와 위험도('{risk_line}')를 명확히 제시.
4) 톤: {tone_guides.get(tone)}
5) 형식: {length_guides.get(length)}
6) AI 결과·약물·기저질환의 연관성을 자연스럽게 설명.
7) 숫자/퍼센트/약물명은 사실적으로 표현.
8) 최종 문장은 반드시 다음 문구로 끝내라:
   "이 설명은 학술제 목적의 예시이며, 실제 진단 및 처방을 대체하지 않습니다."
"""

    instructions = """
[작성 지침]
1. 단계(예: 매우 경도/경도/중등도/정상)의 임상적 의미를 쉬운 말로 요약.
2. 약물(권장/주의/피함)과 기저질환 사이의 연관성을 구체적으로 연결.
3. 권장 약물의 작용기전·효과·대표 부작용을 간단히.
4. ‘주의’/‘피함’ 사유를 근거 위주로 간명하게.
5. 생활 조언(복약/활동/모니터링)을 1~2문장 포함.
6. 임신 관련 항목이 있으면 별도로 언급.
7. 과도한 의료 주장/진단 단정 표현은 피하고, 안내/권고 톤 유지.
"""

    task = f"""
위 [참고데이터]를 바탕으로 {name}님에게 설명을 작성하라.
{('' if not preg_block else preg_block)}
!!! [참고데이터]를 그대로 복사하지 말고, 자연스럽고 읽기 쉬운 문장으로 재구성하라.
"""

    return f"{context}\n{hard_rules}\n{instructions}\n{task}"


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
# ===================== LLM 톤/길이 매핑 & 기본값 =====================
# UI 라벨 ↔ 내부 코드값
TONE_OPTIONS = [("친절하게", "Kind"), ("중립적", "Neutral"), ("전문적", "Expertise")]
LENGTH_OPTIONS = [("짧게", "Short"), ("보통", "Normal"), ("길게", "Detail")]

def _index_of_internal(options, internal_value, fallback=0):
    for i, (_, code) in enumerate(options):
        if code == internal_value:
            return i
    return fallback

def infer_defaults_from_age_simple(age):
    """
    60세 이상: 친절/짧게  (Kind/Short)
    그 외:    중립/보통  (Neutral/Normal)
    """
    try:
        age = int(age)
    except Exception:
        age = None
    if age is not None and age >= 60:
        return ("Kind", "Short")
    return ("Neutral", "Normal")

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

    # 개인화 약물 플랜
    stage = res.get("stage", "NonDemented")
    diseases = info.get("기저질환", []) or []
    plan = personalize_drugs(stage, diseases)

    # 1) 연령 기반 기본값(간단 규칙)
    age = info.get("나이", None)
    default_tone_code, default_length_code = infer_defaults_from_age_simple(age)

    # 2) 자동/수동 토글
    manual = st.toggle("사용자 설정 직접 선택", value=False, help="끄면 연령에 따라 자동으로 톤/길이를 설정합니다.")

    # 3) 한국어 UI 라벨
    tone_labels = [lbl for (lbl, _) in TONE_OPTIONS]
    length_labels = [lbl for (lbl, _) in LENGTH_OPTIONS]

    # 4) 기본 인덱스
    tone_default_idx = _index_of_internal(TONE_OPTIONS, default_tone_code, fallback=1)
    length_default_idx = _index_of_internal(LENGTH_OPTIONS, default_length_code, fallback=1)

    # 5) 선택 UI (언어 선택 제거, 항상 한국어)
    col1, col2 = st.columns(2)
    with col1:
        tone_ui = st.selectbox(
            "톤",
            tone_labels,
            index=tone_default_idx,
            disabled=not manual,
        )
    with col2:
        length_ui = st.selectbox(
            "길이",
            length_labels,
            index=length_default_idx,
            disabled=not manual
        )

    # 6) 내부 코드 확정
    if manual:
        # 한국어 라벨 → 내부 코드값
        tone_code = dict(TONE_OPTIONS)[tone_ui]
        length_code = dict(LENGTH_OPTIONS)[length_ui]
    else:
        tone_code, length_code = default_tone_code, default_length_code

    st.caption(f"현재 설정 · 톤: **{tone_code}** / 길이: **{length_code}** / 언어: **한국어**")

    if st.button("LLM 설명하기"):
        with st.spinner("설명 생성 중..."):
            client = get_openai_client()
            text = generate_llm_explanation(
                client,
                info,
                res,
                plan,
                tone=tone_code,       # "Kind/Neutral/Expertise"
                length=length_code,   # "Short/Normal/Detail"
                language="한국어"     # 한국어 고정
            )
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