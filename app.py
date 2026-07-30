import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Setup
st.set_page_config(page_title="Know Your Worth", layout="wide")

# Custom CSS for Blue Header Bar
st.markdown("""
<style>
    .main-header {
        background-color: #2F98C3;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def train_model():
    np.random.seed(42)
    n = 400
    exp = np.random.uniform(0.5, 15.0, n)
    edu = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    skills = np.random.randint(1, 9, n)
    hub = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

    salary = (50000 + (exp * 8500) + (edu * 12000) + 
              (skills * 3500) + (hub * 20000) + 
              np.random.normal(0, 7500, n))

    df = pd.DataFrame({
        'YearsExperience': exp, 
        'EducationLevel': edu, 
        'SkillsCount': skills, 
        'IsTechHub': hub, 
        'Salary': salary
    })
    
    X = df[['YearsExperience', 'EducationLevel', 'SkillsCount', 'IsTechHub']]
    y = df['Salary']
    model = LinearRegression().fit(X, y)
    
    exp_sorted = np.sort(exp)
    trend = 50000 + (exp_sorted * 8500)
    chart_df = pd.DataFrame({'Experience': exp_sorted, 'Estimated Salary Trend': trend})
    
    return model, chart_df

model, chart_df = train_model()

# Blue Header
st.markdown("<div class='main-header'>Know Your Worth</div>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("User Profile")
job_title = st.sidebar.text_input("Job Title", value="Software Engineer")
edu_label = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
edu_val = {"Bachelor's": 0, "Master's": 1, "PhD": 2}[edu_label]
skills = st.sidebar.slider("Technical Skills Count", 1, 10, 5)
is_hub = st.sidebar.checkbox("Located in Major Tech Hub")
hub_val = 1 if is_hub else 0
exp = st.sidebar.slider("Years of Experience", 0.0, 20.0, 5.0, 0.5)

# Calculate Prediction
pred_salary = model.predict([[exp, edu_val, skills, hub_val]])[0]

# UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Estimated Salary", f"${pred_salary:,.0f}")
    st.write(f"💼 **Title:** {job_title}")
    st.write(f"🎓 **Education:** {edu_label}")
    st.write(f"📍 **Location:** {'Tech Hub' if is_hub else 'Standard'}")
    st.write(f"🕒 **Experience:** {exp} years")

with col2:
    st.subheader("Salary Growth Trend")
    st.line_chart(chart_df, x='Experience', y='Estimated Salary Trend')

st.markdown("---")
st.subheader("How you compare")
st.write("This prediction represents estimated compensation based on profile parameters across technical roles.")
