# 1. Install dependencies
!pip install streamlit scikit-learn pandas numpy matplotlib seaborn -q

# 2. Write app.py
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Tech Salary Estimator", page_icon="💼", layout="wide")
st.title("💼 Tech Salary Estimator")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 400
    exp = np.random.uniform(0.5, 15.0, n)
    edu = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    skills = np.random.randint(1, 9, n)
    hub = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    salary = 50000 + (exp * 8500) + (edu * 12000) + (skills * 3500) + (hub * 20000) + np.random.normal(0, 7500, n)
    return pd.DataFrame({'YearsExperience': exp, 'EducationLevel': edu, 'SkillsCount': skills, 'IsTechHub': hub, 'Salary': salary})

df = load_data()

X = df[['YearsExperience', 'EducationLevel', 'SkillsCount', 'IsTechHub']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

simple_model = LinearRegression().fit(X_train[['YearsExperience']], y_train)
multi_model = LinearRegression().fit(X_train, y_train)

st.header("1. Model Performance")
c1, c2 = st.columns(2)
c1.metric("Simple Model R²", f"{r2_score(y_test, simple_model.predict(X_test[['YearsExperience']])):.3f}")
c2.metric("Multiple Model R²", f"{r2_score(y_test, multi_model.predict(X_test)):.3f}")

st.header("2. Live Predictor")
exp = st.slider("Years of Experience", 0.0, 20.0, 5.0)
edu_label = st.radio("Education", ["Bachelor's", "Master's", "PhD"])
edu = {"Bachelor's": 0, "Master's": 1, "PhD": 2}[edu_label]
skills = st.slider("Skills Count", 1, 10, 4)
hub_label = st.radio("Location", ["Standard", "Tech Hub"])
hub = {"Standard": 0, "Tech Hub": 1}[hub_label]

pred_m = multi_model.predict([[exp, edu, skills, hub]])[0]
st.subheader(f"Estimated Salary: **${pred_m:,.2f}**")
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py successfully created!")
import time
import subprocess
import signal

# 1. Kill old instances (important for Colab)
try:
    print("Stopping previous instances...")
    taskkill = subprocess.Popen(["taskkill", "/F", "/IM", "cloudflared.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    taskkill.wait()
    time.sleep(2)
except Exception:
    pass

# 2. Check if cloudflared is downloaded
try:
    with open("cloudflared", "r"):
        pass
except FileNotFoundError:
    print("Downloading Cloudflare tunnel tool...")
    !wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x cloudflared

# 3. Run the updated Streamlit app in background
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(3)

# 4. Start Cloudflare tunnel and print the new link
print("🚀 Launching Simplified App with Cloudflare Tunnel...\nLink will appear below:\n")
!./cloudflared tunnel --url http://localhost:8501 2>&1 | grep -o 'https://.*\.trycloudflare\.com'
