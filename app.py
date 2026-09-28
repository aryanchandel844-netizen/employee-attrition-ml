
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Page settings
# -----------------------------

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👨‍💼",
    layout="wide"
)


# -----------------------------
# Load model and data
# -----------------------------

model = joblib.load("attrition_model.pkl")

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

training_data = pd.read_csv(
    "X_train.csv"
)


# -----------------------------
# Header
# -----------------------------

st.title("👨‍💼 Employee Attrition Prediction")
st.write(
    "AI/ML powered employee attrition prediction and HR analytics dashboard."
)

st.divider()


# -----------------------------
# KPI calculations
# -----------------------------

total_employees = len(df)

employees_left = len(
    df[df["Attrition"] == "Yes"]
)

employees_stayed = len(
    df[df["Attrition"] == "No"]
)

attrition_rate = (
    employees_left / total_employees
) * 100


# -----------------------------
# KPI Cards
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Employees",
        total_employees
    )

with col2:
    st.metric(
        "Employees Left",
        employees_left
    )

with col3:
    st.metric(
        "Employees Stayed",
        employees_stayed
    )

with col4:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )


st.divider()


# -----------------------------
# Analytics
# -----------------------------

st.subheader("📊 HR Analytics Dashboard")


col1, col2 = st.columns(2)


# -----------------------------
# Pie Chart
# -----------------------------

with col1:

    st.markdown("### 🥧 Attrition Distribution")

    attrition_counts = (
        df["Attrition"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.pie(
        attrition_counts.values,
        labels=attrition_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Employee Attrition Distribution"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )


# -----------------------------
# Department Bar Chart
# -----------------------------

with col2:

    st.markdown("### 📊 Department-wise Employees")

    department_counts = (
        df["Department"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.barplot(
        x=department_counts.index,
        y=department_counts.values,
        ax=ax
    )

    ax.set_xlabel("Department")
    ax.set_ylabel("Employees")
    ax.set_title(
        "Employees by Department"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    st.pyplot(
        fig,
        use_container_width=True
    )


# -----------------------------
# Overtime Chart
# -----------------------------

st.markdown("### 📈 Overtime vs Attrition")

overtime_data = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

overtime_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Overtime")
ax.set_ylabel("Number of Employees")
ax.set_title(
    "Overtime and Employee Attrition"
)

st.pyplot(
    fig,
    use_container_width=True
)


st.divider()


# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("🎯 Employee Attrition Prediction")

st.write(
    "Enter employee information to estimate attrition risk."
)


input_data = {}


# -----------------------------
# Create input fields
# -----------------------------

for column in training_data.columns:

    if training_data[column].dtype == "object":

        options = sorted(
            training_data[column]
            .dropna()
            .unique()
            .tolist()
        )

        input_data[column] = st.selectbox(
            column,
            options
        )

    else:

        min_value = float(
            training_data[column].min()
        )

        max_value = float(
            training_data[column].max()
        )

        default_value = float(
            training_data[column].median()
        )

        input_data[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=default_value
        )


st.divider()


# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "🔮 Predict Attrition",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]


    st.subheader("Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "🔴 High Attrition Risk"
            )

        else:

            st.success(
                "🟢 Low Attrition Risk"
            )


    with result_col2:

        st.metric(
            "Attrition Probability",
            f"{probability * 100:.2f}%"
        )


    st.markdown(
        "### Risk Probability"
    )

    st.progress(
        float(probability)
    )


    if probability >= 0.70:

        st.warning(
            "The model estimates a high probability of employee attrition."
        )

    elif probability >= 0.40:

        st.info(
            "The model estimates a moderate probability of employee attrition."
        )

    else:

        st.success(
            "The model estimates a relatively low probability of employee attrition."
        )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Employee Attrition Prediction System | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
