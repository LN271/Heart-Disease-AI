import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai

# 1. جلب المفتاح السري من إعدادات Streamlit
api_key = st.secrets["GEMINI_API_KEY"]

# 2. إعداد مكتبة جيميناي
genai.configure(api_key=api_key)

# 3. تعليمات صارمة للذكاء الاصطناعي (كنص عادي)
SYSTEM_INSTRUCTION = """
أنت طبيب قلب استشاري. سأعطيك 'نتائج التحليل الطبي' للمريض. 
اشرحها للمريض بأسلوب إنساني، مبسط، ومطمئن، وقدم له نصائح وقائية وعلاجية. 
ممنوع تماماً استخدام كلمات مثل 'خوارزمية'، 'ذكاء اصطناعي'، 'نموذج'، أو 'برنامج'. 
تحدث دائماً كطبيب بشري حقيقي يقرأ تقريراً طبياً لمريضه.
انت طبيب يتكلم عن القلب فقط
واذا طلب صورة او مخطط بياني للتوضيح عن سبب النتائج ارسال له 
"""

# 4. تجهيز نموذج الذكاء الاصطناعي وإرفاق التعليمات به (سميناه model_ai لتجنب التداخل)
model_ai = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_INSTRUCTION)
# --- إعدادات صفحة الويب/الموبايل ---
st.set_page_config(page_title="HeartShield AI", page_icon="🫀", layout="centered")

# --- إخفاء أزرار Streamlit العلوية (Deploy, GitHub, Menu) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp > header {display: none;}
            .stDeployButton {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# دالة لتحميل ملفات التحليل (سميناه ml_model لتجنب التداخل)
@st.cache_resource
def load_models():
    ml_model = joblib.load('heart_attack_stack_model.pkl')
    scaler = joblib.load('scaler.pkl')
    best_threshold = joblib.load('best_threshold.pkl')
    return ml_model, scaler, best_threshold

try:
    ml_model, scaler, best_threshold = load_models()
except Exception as e:
    st.error(f"⚠️ فشل تحميل ملفات التحليل: {e}")
    st.stop()

# --- إدارة حالة التطبيق (Session State) ---
if 'last_status' not in st.session_state:
    st.session_state.last_status = None
if 'last_prob' not in st.session_state:
    st.session_state.last_prob = None
if 'last_data' not in st.session_state:
    st.session_state.last_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- تصميم الواجهة ---
st.markdown("<h1 style='text-align: center; color: #58A6FF;'>نظام التشخيص المتكامل 🫀</h1>", unsafe_allow_html=True)

# نظام التبويبات
tab1, tab2 = st.tabs(["📊 الفحص الطبي", "🩺 الاستشاري الذكي"])

with tab1:
    st.subheader("إدخال بيانات المريض")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("👤 العمر (Age)", min_value=1, max_value=120, value=45)
        total_cholesterol = st.number_input("🧪 الكوليسترول الكلي", min_value=50, max_value=600, value=195)
        systolic_bp = st.number_input("🩸 الضغط الانقباضي", min_value=70, max_value=250, value=125)
        hdl = st.number_input("✨ الكوليسترول النافع (HDL)", min_value=10, max_value=150, value=52)
        
    with col2:
        sex_input = st.selectbox("⚥ الجنس", ["ذكر", "أنثى"])
        sex = 1.0 if sex_input == "ذكر" else 0.0
        
        smoking_input = st.selectbox("🚬 التدخين", ["لا", "نعم"])
        smoking = 1.0 if smoking_input == "نعم" else 0.0
        
        diabetes_input = st.selectbox("🍬 السكري", ["لا", "نعم"])
        diabetes = 1.0 if diabetes_input == "نعم" else 0.0
        
        diastolic_bp = st.number_input("📉 الضغط الانبساطي", min_value=40, max_value=150, value=82)
        
    ldl = st.number_input("⚠️ الكوليسترول الضار (LDL)", min_value=30, max_value=300, value=118)

    if st.button(" إصدار نتائج التحليل ", use_container_width=True, type="primary"):
        data = {
            "age": float(age),
            "sex": sex,
            "total_cholesterol": float(total_cholesterol),
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "smoking": smoking,
            "diabetes": diabetes,
            "hdl": float(hdl),
            "ldl": float(ldl)
        }
        
        df = pd.DataFrame([data])
        
        try:
            correct_order = scaler.feature_names_in_
            df = df[correct_order]
        except AttributeError:
            try:
                correct_order = ml_model.feature_names_in_
                df = df[correct_order]
            except AttributeError:
                expected_columns = ['age', 'sex', 'total_cholesterol', 'systolic_bp', 'diastolic_bp', 'smoking', 'diabetes', 'hdl', 'ldl']
                df = df[expected_columns]
            
        scaled = scaler.transform(df)
        prob = ml_model.predict_proba(scaled)[:, 1][0]
        
        is_infected = prob >= best_threshold
        status = "🚨 حالة حرجة (تحتاج متابعة)" if is_infected else "✅ حالة سليمة (مؤشرات طبيعية)"
        
        st.session_state.last_status = status
        st.session_state.last_prob = prob
        st.session_state.last_data = data
        
        # عرض النتيجة
        if is_infected:
            st.error(f"**النتيجة الأولية:** {status} | **مؤشر الخطر:** {prob*100:.1f}%")
        else:
            st.success(f"**النتيجة الأولية:** {status} | **مؤشر الخطر:** {prob*100:.1f}%")
            
        # الاتصال التلقائي بالطبيب الذكي (بدون تدخل المستخدم)
        with st.spinner('⏳ جاري إرسال النتائج للاستشاري لتحليلها وإعداد التقرير...'):
            prompt = f"نتائج التحليل الطبي أظهرت أن المريض في: {status} بمؤشر خطر {prob*100:.1f}%. المعطيات الحيوية للمريض هي: {data}. بصفتك طبيب قلب، اشرح لي هذه النتيجة وقدم لي نصائح طبية."
            
            # مسح المحادثة القديمة وبدء واحدة جديدة
            st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "user", "text": "مرحباً دكتور، هذه نتائج التحليل الطبي الخاصة بي، أرجو الاطلاع عليها وتوضيح حالتي."})
            
            try:
                # إرسال البيانات للذكاء الاصطناعي
                response = model_ai.generate_content(prompt)
                st.session_state.chat_history.append({"role": "ai", "text": response.text})
                st.success("✅ الطبيب قام بدراسة النتائج! انتقل إلى  '🩺 الاستشاري الذكي' بالأعلى لقراءة التقرير والتحدث معه.")
            except Exception as e:
                st.error(f"⚠️ حدث خطأ في الاتصال: {e}")

with tab2:
    st.subheader("محادثة مع الطبيب الاستشاري")
    
    if len(st.session_state.chat_history) == 0:
        st.info("👈 يرجى إدخال البيانات في التبويب الأول والضغط على 'إصدار نتائج التحليل' لكي يقوم الطبيب بدراسة حالتك.")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["text"])
            else:
                st.chat_message("assistant", avatar="🩺").write(msg["text"])
                
        if user_input := st.chat_input("تحدث مع الطبيب هنا..."):
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            st.chat_message("user").write(user_input)
            
            with st.spinner('الطبيب يكتب...'):
                try:
                    # إرسال استفسار المريض للذكاء الاصطناعي
                    response = model_ai.generate_content(user_input)
                    st.session_state.chat_history.append({"role": "ai", "text": response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"حدث خطأ: {e}")
