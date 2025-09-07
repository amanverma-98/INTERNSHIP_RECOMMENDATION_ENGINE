import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------- Styling -------------------------
st.markdown("""
<style>
.stApp {
    background: url("https://img.youtube.com/vi/OmiDv60ah84/maxresdefault.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    position: relative;
}
.stApp::before {
    content: "";
    position: absolute;
    top:0; left:0; right:0; bottom:0;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    z-index:0;
}
.stApp > div { position: relative; z-index: 1; }
.recommend-card {
    background: rgba(255,255,255,0.95);
    color:#000;
    border-radius:15px;
    padding:25px;
    margin-bottom:20px;
    box-shadow:0px 6px 15px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.recommend-card:hover {
    transform: translateY(-4px);
    box-shadow:0px 10px 20px rgba(0,0,0,0.5);
}
h1,h2,h3 {
    color:#ffeb3b;
    font-weight:bold;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
}
.stButton>button {
    background: linear-gradient(90deg,#ff9800,#ff5722);
    color:white;
    border-radius:12px;
    padding:10px 24px;
    font-weight:bold;
    border:none;
    box-shadow:2px 4px 12px rgba(0,0,0,0.3);
    transition: background 0.3s ease, transform 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#ff5722,#e64a19);
    transform: scale(1.05);
}
.stCaption { color:#f0f0f0; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Page Config -------------------------
st.set_page_config(page_title="PM Internship Recommender", page_icon="🧭", layout="centered")

# ------------------------- Load Data -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./data/internships.csv")
    
    # Split English / Hindi columns
    for col in ["title", "skills", "sector", "location"]:
        df[f"{col}_en"] = df[col].apply(lambda x: x.split(" / ")[0].strip())
        df[f"{col}_hi"] = df[col].apply(lambda x: x.split(" / ")[1].strip() if " / " in x else x.strip())
    
    # Education mapping for Hindi
    edu_mapping = {"UG": "स्नातक", "PG": "स्नातकोत्तर"}
    df["education_hi"] = df["education_level"].map(edu_mapping).fillna(df["education_level"])
    
    # Preprocess lists for vectorization
    df["skills_list_en"] = df["skills_en"].apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    df["skills_list_hi"] = df["skills_hi"].apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    df["sector_list_en"] = df["sector_en"].apply(lambda s: [s])
    df["sector_list_hi"] = df["sector_hi"].apply(lambda s: [s])
    df["location_list_en"] = df["location_en"].apply(lambda s: [s])
    df["location_list_hi"] = df["location_hi"].apply(lambda s: [s])
    df["edu_list_en"] = df["education_level"].apply(lambda s: [s])
    df["edu_list_hi"] = df["education_hi"].apply(lambda s: [s])
    
    return df

df = load_data()

# ------------------------- Language Toggle -------------------------
LANG = st.toggle("हिन्दी / English", value=False)
def t(en, hi):
    return hi if LANG else en

# ------------------------- Build Vocabularies -------------------------
skills_vocab = sorted({s for row in (df["skills_list_hi"] if LANG else df["skills_list_en"]) for s in row})
sector_vocab = sorted({s[0] for s in (df["sector_list_hi"] if LANG else df["sector_list_en"])})
location_vocab = sorted({s[0] for s in (df["location_list_hi"] if LANG else df["location_list_en"])})
edu_vocab = sorted({s[0] for s in (df["edu_list_hi"] if LANG else df["edu_list_en"])})

# ------------------------- Vectorizers -------------------------
mlb_skills = MultiLabelBinarizer(classes=skills_vocab)
mlb_sector = MultiLabelBinarizer(classes=sector_vocab)
mlb_location = MultiLabelBinarizer(classes=location_vocab)
mlb_edu = MultiLabelBinarizer(classes=edu_vocab)

skills_enc = mlb_skills.fit_transform(df["skills_list_hi"] if LANG else df["skills_list_en"])
sector_enc = mlb_sector.fit_transform(df["sector_list_hi"] if LANG else df["sector_list_en"])
location_enc = mlb_location.fit_transform(df["location_list_hi"] if LANG else df["location_list_en"])
edu_enc = mlb_edu.fit_transform(df["edu_list_hi"] if LANG else df["edu_list_en"])

internship_vectors = np.hstack([skills_enc, sector_enc, location_enc, edu_enc])

# ------------------------- UI Header -------------------------
st.markdown(f"<h1 style='text-align:center;'>🧭 {t('PM Internship Recommendation Engine','PM इंटर्नशिप सिफारिश इंजन')}</h1>", unsafe_allow_html=True)
st.caption(t(
    "🌐 Lightweight, mobile-friendly AI tool for students across India",
    "🌐 छात्रों के लिए हल्का, मोबाइल-फ्रेंडली एआई टूल (संपूर्ण भारत में)"
))

# ------------------------- Profile Form -------------------------
with st.form("profile"):
    st.subheader(t("Your Profile", "आपकी प्रोफ़ाइल"))
    c1, c2 = st.columns(2, vertical_alignment="center")
    
    with c1:
        selected_skills = st.multiselect("🛠 " + t("Select your skills","अपनी स्किल्स चुनें"), options=skills_vocab)
        selected_sector = st.selectbox("🏷 " + t("Sector of interest","रुचि का सेक्टर"), options=[""] + sector_vocab, index=0)
    
    with c2:
        selected_location = st.selectbox("📍 " + t("Preferred location","पसंदीदा स्थान"), options=[""] + location_vocab, index=0)
        selected_edu = st.selectbox("🎓 " + t("Education level","शिक्षा स्तर"), options=[""] + edu_vocab, index=0)
    
    top_k = st.slider(t("How many recommendations?", "कितनी सिफारिशें चाहिए?"), 3, 5, 5)
    submitted = st.form_submit_button(t("Get Recommendations","सिफारिशें देखें"))

# ------------------------- Candidate Encoding -------------------------
def encode_candidate(skills, sector, location, edu):
    skills_list = skills if skills else []
    sector_list = [sector] if sector else []
    location_list = [location] if location else []
    edu_list = [edu] if edu else []

    vec = np.hstack([
        mlb_skills.transform([skills_list]),
        mlb_sector.transform([sector_list]),
        mlb_location.transform([location_list]),
        mlb_edu.transform([edu_list]),
    ])
    return vec

# ------------------------- Match Breakdown -------------------------
def match_breakdown(c_row, cand_skills):
    intern_skills = set(c_row["skills_list_hi"] if LANG else c_row["skills_list_en"])
    overlap = sorted(intern_skills.intersection(set(cand_skills)))
    return overlap, len(overlap), len(intern_skills)

# ------------------------- Recommendations -------------------------
if submitted:
    if not selected_skills and not selected_sector and not selected_location and not selected_edu:
        st.info(t("Please select at least one input.","कृपया कम से कम एक विकल्प चुनें।"))
    else:
        cand_vec = encode_candidate(selected_skills, selected_sector, selected_location, selected_edu)
        sims = cosine_similarity(cand_vec, internship_vectors)[0]
        order = np.argsort(-sims)[:top_k]

        st.markdown("---")
        st.subheader(t("Top Recommendations","शीर्ष सिफारिशें"))

for rank, idx in enumerate(order, start=1):
    row = df.iloc[idx]
    overlap, n_overlap, n_total = match_breakdown(row, selected_skills)

    st.markdown(f"""
    <div class="recommend-card">
    <h3 style="color:#1565c0;">{rank}. {row['title']}</h3>
    <p><b>📍 {t("Location","स्थान")}:</b> {row['location']}</p>
    <p><b>🏷 {t("Sector","क्षेत्र")}:</b> {row['sector']}</p>
    <p><b>🎓 {t("Education","शिक्षा")}:</b> {row['education_level']}</p>
    <p><b>🛠 {t("Skills","कौशल")}:</b> {", ".join(row["skills_list"])}</p>
    <p><b>✅ {t("Your Overlap","आपकी मेल स्किल्स")}:</b> {", ".join(overlap) if overlap else t("No direct overlap","कोई मेल नहीं")}</p>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Add Apply button here with correct indentation
    st.markdown(f"""
    <a href="https://pminternship.mca.gov.in/" target="_blank">
       <div style="display:inline-block; background-color:#1565c0; color:white;
           padding:8px 16px; border:none; border-radius:6px; cursor:pointer;
           text-align:center; text-decoration:none; font-weight:bold;">
           {t("Apply on PM Portal","पीएम पोर्टल पर आवेदन करें")}
       </div>
    </a>
    """, unsafe_allow_html=True)



    st.caption(t("Tip: Add more skills or change location to improve matches.","टिप: बेहतर मेल के लिए अधिक स्किल्स जोड़ें या स्थान बदलें."))

# ------------------------- About -------------------------
st.markdown("---")
with st.expander(t("About this prototype","प्रोटोटाइप के बारे में")):
    st.write(t(
        "We use one-hot vectors for skills, sector, location, education and cosine similarity to rank internships. Designed to be lightweight and easy to integrate.",
        "हम स्किल्स, सेक्टर, स्थान और शिक्षा के लिए वन-हॉट वेक्टर और कोसाइन समानता का उपयोग करते हैं। यह हल्का है और एकीकरण में आसान है।"
    ))
