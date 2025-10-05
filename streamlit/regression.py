import streamlit as st
import numpy as np 
from sklearn.linear_model import LinearRegression
X= np.array([[1],[2],[3],[4],[5]])
Y=np.array([40,50,60,70,80])
model=LinearRegression().fit(X,Y)
st.title("Simple Supervised Regression Demo")
st.write("""This demo shows a **Linear Regression** model trained on a tiny dataset
of *study hours* vs *test scores*.

**How it works:**
1. We trained a model on 5 data points (study hours → test scores)
2. Use the slider to input different study hours
3. The model predicts what test score you'd get
4. The model parameters show the mathematical relationship it learned
""")
hours=st.slider(
    "Hours studied",
    min_value=0.0,
    max_value=10.0,
    step=0.5
)
predicted_score = model.predict([[hours]])[0]
st.write(f"**Predicted score for {hours} hours of study:** {predicted_score:.2f}")
st.caption(f"Slope(coef):{model.coef_[0]:.2f},Intercept: {model.intercept_:.2f}")