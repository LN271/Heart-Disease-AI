import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai

# --- 1. الإعدادات الفنية ومفتاح الـ API ---
# تأكد أنك كتبت MY_API_KEY داخل Secrets في لوحة تحكم Streamlit
try:
    api_key = st.secrets["MY_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("⚠️ لم يتم العثور على مفتاح 'MY_API_KEY' في إعدادات Secrets.")
    st.stop()

# تعليمات الطبيب الاستشاري الصارمة
SYSTEM_INSTRUCTION = """
أنت طبيب قلب استشاري. سأعطيك 'نتائج التحليل الطبي' للمريض. 
اشرحها للمريض بأسلوب إنساني، مبسط، ومطمئن، وقدم له نصائح وقائية وعلاجية. 
ممنوع تماماً استخدام كلمات مثل 'خوارزمية'، 'ذكاء اصطناعي'، 'نموذج'، أو 'برنامج'. 
تحدث دائماً كطبيب بشري حقيقي يقرأ تقريراً طبياً لمريضه.
أنت طبيب يتكلم عن القلب فقط.
وإذا طلب صورة أو مخطط بياني للتوضيح عن سبب النتائج، قدم له الوصف الطبي الدقيق.
"""

# --- 2. إعدادات الصفحة والواجهة ---
st.set_page_config(page_title="HeartShield AI", page_icon="🫀", layout="centered")

# إخفاء عناصر Streamlit لزيادة احترافية التطبيق
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

# --- 3. تحميل ملفات التحليل الذكي (.pkl) ---
@st.cache_resource
def load_models():
    # تأكد أن هذه الملفات مرفوعة بجانب ملف app.py على GitHub
    model = joblib.load('heart_attack_stack_model.pkl')
    scaler = joblib.load('scaler.pkl')
    best_threshold = joblib.load('best_threshold.pkl')
    return model, scaler, best_threshold

try:
    model, scaler, best_threshold = load_models()
except Exception as e:
    st.error(f"⚠️ فشل تحميل ملفات التحليل: {e}")
    st.stop()

# إدارة ذاكرة المحادثة
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 4. تصميم الواجهة الرسومية ---
st.markdown("<h1 style='text-align: center; color: #58A6FF;'>نظام التشخيص المتكامل 🫀</h1>", unsafe_allow_html=True)

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

    if st.button("إصدار نتائج التحليل", use_container_width=True, type="primary"):
        data = {
            "age": int(age), "sex": sex, "total_cholesterol": float(total_cholesterol),
            "systolic_bp": float(systolic_bp), "diastolic_bp": float(diastolic_bp),
            "smoking": smoking, "diabetes": diabetes, "hdl": float(hdl), "ldl": float(ldl)
        }
        
        df = pd.DataFrame([data])
        
        try:
            if hasattr(scaler, 'feature_names_in_'):
                correct_order = scaler.feature_names_in_
            elif hasattr(model, 'feature_names_in_'):
                correct_order = model.feature_names_in_
            else:
                correct_order = ['age', 'sex', 'total_cholesterol', 'systolic_bp', 'diastolic_bp', 'smoking', 'diabetes', 'hdl', 'ldl']
            
            df = df[correct_order]
            scaled_data = scaler.transform(df)
            prob = model.predict_proba(scaled_data)[:, 1][0]
            
            is_infected = prob >= best_threshold
            status = "🚨 حالة حرجة (تحتاج متابعة)" if is_infected else "✅ حالة سليمة (مؤشرات طبيعية)"
            
            if is_infected:
                st.error(f"النتيجة الأولية: {status} | مؤشر الخطر: {prob*100:.1f}%")
            else:
                st.success(f"النتيجة الأولية: {status} | مؤشر الخطر: {prob*100:.1f}%")
            
            with st.spinner('⏳ جاري إرسال النتائج للاستشاري لتحليلها...'):
                prompt = f"نتائج التحليل الطبي أظهرت أن المريض في: {status} بمؤشر خطر {prob*100:.1f}%. المعطيات الحيوية للمريض هي: {data}. بصفتك طبيب قلب، اشرح لي هذه النتيجة وقدم لي نصائح طبية."
                
                st.session_state.chat_history = [
                    {"role": "user", "parts": ["مرحباً دكتور، هذه نتائج التحليل الطبي الخاصة بي، أرجو الاطلاع عليها وتوضيح حالتي."]}
                ]
                
                # تم تصحيح اسم الموديل هنا إلى gemini-1.5-flash
                chat_model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash', 
                    system_instruction=SYSTEM_INSTRUCTION
                )
                
                response = chat_model.generate_content(prompt)
                st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
                st.success("✅ الطبيب قام بدراسة النتائج! انتقل لتبويب 'الاستشاري الذكي' لقراءة التقرير.")
                
        except Exception as e:
            st.error(f"⚠️ خطأ في معالجة البيانات: {e}")

with tab2:
    st.subheader("محادثة مع الطبيب الاستشاري")
    
    if not st.session_state.chat_history:
        st.info("👈 يرجى إدخال البيانات في التبويب الأول والضغط على 'إصدار نتائج التحليل' أولاً.")
    else:
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            avatar = "🩺" if role == "assistant" else None
            st.chat_message(role, avatar=avatar).write(msg["parts"][0])
                
        if user_input := st.chat_input("اسأل الطبيب أي سؤال إضافي..."):
            st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
            st.chat_message("user").write(user_input)
            
            with st.spinner('الطبيب يفكر...'):
                try:
                    # تم تصحيح اسم الموديل هنا أيضاً
                    chat_model = genai.GenerativeModel(
                        model_name='gemini-1.5-flash',
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                    response = chat_model.generate_content(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
                    st.rerun()
                except Exception as e:
                    st.error(f"حدث خطأ في الاتصال: {e}")
