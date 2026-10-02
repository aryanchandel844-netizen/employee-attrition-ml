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
# Analytics Dashboard
# -----------------------------

st.subheader("📊 HR Analytics Dashboard")

col1, col2 = st.columns(2)


# -----------------------------
# Attrition Distribution
# -----------------------------

with col1:

    st.markdown("### 🥧 Attrition Distribution")

    attrition_counts = df["Attrition"].value_counts()

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

    plt.close(fig)


# -----------------------------
# Department Analysis
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

    plt.close(fig)


# -----------------------------
# Overtime Analysis
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

plt.close(fig)


st.divider()


# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("🎯 Employee Attrition Prediction")

st.write(
    "Enter employee information to estimate attrition risk."
)


# -----------------------------
# Input data
# -----------------------------

input_data = {}


# These columns contain text/categorical values
categorical_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]


for column in training_data.columns:

    # -------------------------
    # Categorical input
    # -------------------------

    if column in categorical_columns:

        options = sorted(
            training_data[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        input_data[column] = st.selectbox(
            column,
            options
        )

    # -------------------------
    # Numeric input
    # -------------------------

    else:

        numeric_values = pd.to_numeric(
            training_data[column],
            errors="coerce"
        ).dropna()

        min_value = int(
            numeric_values.min()
        )

        max_value = int(
            numeric_values.max()
        )

        default_value = int(
            numeric_values.median()
        )

        input_data[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=1
        )


st.divider()


# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "🔮 Predict Attrition",
    use_container_width=True
):

    # Create DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )

    # Make prediction
    prediction = model.predict(
        input_df
    )[0]

    # Get probability
    probability = model.predict_proba(
        input_df
    )[0][1]


    # -------------------------
    # Prediction Result
    # -------------------------

    st.subheader(
        "Prediction Result"
    )


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


    # -------------------------
    # Probability
    # -------------------------

    st.markdown(
        "### Risk Probability"
    )

    st.progress(
        float(probability)
    )


    # -------------------------
    # Risk message
    # -------------------------

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
