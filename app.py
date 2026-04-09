import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from src.detectors.chain_master import VisageIQ_Engine

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="VisageIQ AI", layout="wide", page_icon="🧠")

# ─────────────────────────────────────────
# UI THEME
# ─────────────────────────────────────────
st.markdown("""
<style>

/* GLOBAL */
html, body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* HERO */
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1e293b, #020617);
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* CARD */
.card {
    background: rgba(30, 41, 59, 0.6);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    border-radius: 12px;
    color: white;
    font-weight: bold;
}

/* INPUT */
.stTextInput input {
    background: #1e293b;
    color: white;
    border-radius: 10px;
}

/* BADGES */
.success {color:#22c55e;}
.warn {color:#facc15;}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* CAMERA MIRROR */
video { transform: scaleX(-1); }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>🧠 VisageIQ AI</h1>
<p>Face Recognition • Mood Detection • Attendance System</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────
@st.cache_resource
def load_engine():
    return VisageIQ_Engine()

engine = load_engine()

# ─────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────
menu = st.sidebar.radio("Navigation", [
    "📷 Live Detection",
    "➕ Register Face",
    "👥 Manage Faces"
])

# ─────────────────────────────────────────
# PROCESS FUNCTION
# ─────────────────────────────────────────
def process(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        faces, rgb = engine.process_frame(frame)
        encodings = engine.get_embeddings(rgb, faces)
        moods = engine.analyze_mood(frame, faces)
    except Exception as e:
        st.error(f"❌ Detection failed: {e}")
        return None, []

    output = frame.copy()
    results = []

    for i, (x,y,w,h) in enumerate(faces):
        name = engine.identify_face(encodings[i]) if i < len(encodings) else "Unknown"
        mood = moods[i] if i < len(moods) else "?"

        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,80),2)
        cv2.putText(output,f"{name} | {mood}",(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,80),2)

        results.append((name, mood))

    return output, results

# ─────────────────────────────────────────
# LIVE DETECTION
# ─────────────────────────────────────────
if menu == "📷 Live Detection":

    st.markdown("### 📷 Live AI Detection")

    img = st.camera_input("Capture Frame")

    if img:
        image = Image.open(img).convert("RGB")
        output, results = process(image)

        if output is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(image, caption="Original", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(output, caption="AI Output", channels="BGR", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if not results:
                st.warning("⚠️ No faces detected.")
            else:
                st.markdown(f"""
                <div class="card">
                ✅ <b>{len(results)}</b> face(s) detected
                </div>
                """, unsafe_allow_html=True)

                for i, (name, mood) in enumerate(results):
                    status = "success" if name != "Unknown" else "warn"

                    st.markdown(f"""
                    <div class="card">
                    <b>Face {i+1}</b><br>
                    👤 Name: <b>{name}</b><br>
                    😊 Mood: <b>{mood}</b><br>
                    📌 Status: <span class="{status}">
                    {'Present' if name!='Unknown' else 'Unknown'}
                    </span><br>
                    ⏱ Time: {datetime.now().strftime("%H:%M:%S")}
                    </div>
                    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────
elif menu == "➕ Register Face":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### ➕ Register Face")

    name = st.text_input("Full Name")

    option = st.radio("Input Method", ["Upload", "Camera"], horizontal=True)

    if option == "Upload":
        file = st.file_uploader("Upload Image", type=["jpg","png"])

        if st.button("Register"):
            if not name:
                st.warning("Enter name")
            elif not file:
                st.warning("Upload image")
            else:
                img = Image.open(file)
                res = engine.register_face(np.array(img), name)

                if res is True:
                    st.success("Registered successfully!")
                elif res == "duplicate":
                    st.warning("Already exists")
                else:
                    st.error("No face detected")

    else:
        cam = st.camera_input("Take Photo")

        if st.button("Register from Camera"):
            if not name:
                st.warning("Enter name")
            elif not cam:
                st.warning("Capture image")
            else:
                img = Image.open(cam)
                res = engine.register_face(np.array(img), name)

                if res is True:
                    st.success("Registered successfully!")
                elif res == "duplicate":
                    st.warning("Already exists")
                else:
                    st.error("No face detected")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# MANAGE
# ─────────────────────────────────────────
elif menu == "👥 Manage Faces":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 👥 Registered Faces")

    names = engine.get_registered_names()

    if not names:
        st.info("No faces registered.")
    else:
        for i, n in enumerate(names):
            col1, col2 = st.columns([4,1])

            with col1:
                st.markdown(f"""
                <div class="card">
                👤 <b>{n}</b>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    idx = engine.known_names.index(n)
                    engine.known_names.pop(idx)
                    engine.known_encodings.pop(idx)
                    engine._save_db()
                    st.success(f"{n} deleted")
                    st.rerun()

    if st.button("⚠️ Clear Database"):
        engine.known_names = []
        engine.known_encodings = []
        engine._save_db()
        st.success("Database cleared")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)