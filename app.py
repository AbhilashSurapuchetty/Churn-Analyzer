import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("📉 Customer Churn Prediction & Model Comparison Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
selected_contract = st.sidebar.multiselect(
    "Contract Type", df['Contract'].unique(), default=df['Contract'].unique())
filtered_df = df[df['Contract'].isin(selected_contract)]

# Show raw data
st.subheader("🔢 Sample Filtered Data")
st.dataframe(filtered_df.head())

# Plot: Churn by Contract Type
st.subheader("📊 Churn Count by Contract Type")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Contract", hue="Churn", ax=ax)
st.pyplot(fig)

# ======== Data Preprocessing ========
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != "customerID":
        df[col] = le.fit_transform(df[col])

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ======== Train Multiple Models ========
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = []
all_predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_predictions[name] = y_pred
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)

# ======== Show Results ========
# ======== Show Results ========
st.subheader("📋 Model Comparison Table")

# Apply formatting only to numeric columns
st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2f}",
        "Precision": "{:.2f}",
        "Recall": "{:.2f}",
        "F1": "{:.2f}"
    }),
    use_container_width=True
)


# ======== Radar Chart ========
st.subheader("🕸️ Radar Plot: Metric Comparison")

normalized_df = results_df.copy()
for metric in ["Accuracy", "Precision", "Recall", "F1"]:
    normalized_df[metric] = normalized_df[metric] / normalized_df[metric].max()

fig = go.Figure()
for i, row in normalized_df.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=row[["Accuracy", "Precision", "Recall", "F1"]].tolist(),
        theta=["Accuracy", "Precision", "Recall", "F1"],
        fill='toself',
        name=row["Model"]
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True
)
st.plotly_chart(fig)

# ======== Feature Importance (from Logistic Regression) ========
st.subheader("📌 Top Features (Logistic Regression)")
logreg = models["Logistic Regression"]
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": logreg.coef_[0]
}).sort_values(by="Importance", key=abs, ascending=False)
st.dataframe(importance.head(10))

# ======== Detailed Report (Top Model) ========
st.subheader("🧾 Classification Report for Best Model")
top_model_name = results_df.iloc[0]["Model"]
top_model_pred = all_predictions[top_model_name]
report = classification_report(y_test, top_model_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
