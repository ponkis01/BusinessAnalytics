import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Konfiguration der Streamlit Seite
# ------------------------------
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📈",
    layout="wide",
)

col1, col2 = st.columns([2, 1])  
with col1:
    st.title("Employee Survival Rate Prediction")
    st.write("""
    Welcome to the Employee Survival Prediction Page! 
    This tool allows you to predict how long an employee will stay at your company based on historical data.
    """)
with col2:
    st.image("https://raw.githubusercontent.com/ponkis01/BusinessAnalytics/refs/heads/main/employee.svg", width=350)


# ------------------------------
# 2. Training Datensatz laden 
# ------------------------------
st.header("📂 Training Dataset")

# URL zur Datei im GitHub-Repository
url = "https://raw.githubusercontent.com/ponkis01/BusinessAnalytics/refs/heads/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
# XLSX-Datei in einen DataFrame laden
train_df = pd.read_csv(url)

# Überprüfung der geladenen Daten
print(train_df.head())


# ------------------------------
# 3. Preprocessing der Daten
# ------------------------------
drop_columns = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'YearsWithCurrManager', 'YearsInCurrentRole']
categorical_columns = [
    'BusinessTravel', 'Department', 'Education', 'Gender',
    'EducationField', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime',
    'StockOptionLevel', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'WorkLifeBalance'
]
numerical_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'YearsAtCompany',
    'JobLevel'  
]
columns_to_encode = [
    'BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender', 
    'JobRole', 'MaritalStatus', 'OverTime', 'Over18',
    'StockOptionLevel', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'WorkLifeBalance'
]
# Wird spezifisch definiert, da sonst ein error auftritt (Training- und Validation-Datensätze müssen diesselbe Reihenfolge der Variablen haben)
feature_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsSinceLastPromotion',
    'YearsAtCompany', 'JobLevel', 'BusinessTravel', 'Department',
    'Education', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
    'OverTime', 'Over18', 'StockOptionLevel', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'WorkLifeBalance'
]

# Preprocessing der Trainingsdaten
def preprocess_training_data(df):
    
    df = df.drop(drop_columns, axis=1)
    
    # Attrition bearbeiten
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        if df['Attrition'].isnull().any():
            st.warning("Some 'Attrition' values could not be mapped to 1/0 and will be dropped.")
            df = df.dropna(subset=['Attrition'])
        df['Attrition'] = df['Attrition'].astype(int)
    
    df[categorical_columns] = df[categorical_columns].astype('category')
    
    # missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Training dataset contains missing values. Filling missing values by dropping rows.")
        df = df.dropna()
    
    # Encoden der kategorischen Variablen
    label_encoder = {}
    for col in columns_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        df[col] = df[col].astype('category')
        label_encoder[col] = le  
    
    return df, label_encoder

with st.spinner("Preprocessing training data..."):
    train_df_processed, label_encoder = preprocess_training_data(train_df)

# ------------------------------
# 4. Modell Training
# ------------------------------
def prepare_survival_data(df):
    event_column = df['Attrition'].astype(bool)  
    time_column = df['YearsAtCompany']
    survival_data = Surv.from_arrays(event=event_column, time=time_column)
    X = df.drop(['Attrition', 'YearsAtCompany'], axis=1)
    y = survival_data
    return X, y

# Survival data vorbereiten 
X_train, y_train = prepare_survival_data(train_df_processed)

# Random Survival Forest Modell trainieren 
with st.spinner("Training the Random Survival Forest model..."):
    try:
        rsf_model = RandomSurvivalForest(
            n_estimators=1000,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        rsf_model.fit(X_train, y_train)
        st.success("Model trained successfully!")
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

# ------------------------------
# 5. Kaplan-Meier Survival Kurve
# ------------------------------
with st.spinner("Generating Kaplan-Meier survival curve..."):
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=train_df_processed['YearsAtCompany'], event_observed=train_df_processed['Attrition'])

    except Exception as e:
        st.error(f"Error generating Kaplan-Meier curve: {e}")

# ------------------------------
# 6. User Input für Prediction (Validation Set)
# ------------------------------
st.header("🔍 Predict Employee Turnover")

st.write("""
Provide the details of the employee you want to analyze. The application will predict the probability of the employee staying with the company over time.
""")

# User Input bekommen
def get_user_input():
    with st.form("employee_form"):
        st.subheader("📝 Enter Employee Details")
        with st.expander("Demographics"):
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Gender = st.selectbox("Gender", options=sorted(train_df['Gender'].unique()))
            MaritalStatus = st.selectbox("Marital Status", options=sorted(train_df['MaritalStatus'].unique()))
            Over18 = st.selectbox("Over 18", options=sorted(train_df['Over18'].unique()))

        with st.expander("Compensation and Benefits"):
            MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
            DailyRate = st.number_input("Daily Rate (max 1499)", min_value=0, value=500)
            HourlyRate = st.number_input("Hourly Rate", min_value=0, value=50)
            MonthlyRate = st.number_input("Monthly Rate", min_value=0, value=5000)
            PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, value=10)
            st.session_state["PercentSalaryHike"] = PercentSalaryHike
            StockOptionLevel = st.selectbox("Stock Option Level", options=sorted(train_df['StockOptionLevel'].unique()))

        with st.expander("Job Information"):
            JobRole = st.selectbox("Job Role", options=sorted(train_df['JobRole'].unique()))
            Department = st.selectbox("Department", options=sorted(train_df['Department'].unique()))
            JobLevel = st.number_input("Job Level", min_value=1, max_value=5, value=1)
            YearsAtCompany = st.number_input("Years at Company", min_value=0, value=3)
            YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, value=2)
            TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, value=1)
            OverTime = st.selectbox("Over Time", options=sorted(train_df['OverTime'].unique()))

        with st.expander("Work Environment Satisfaction"):

            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", options=sorted(train_df['EnvironmentSatisfaction'].unique()))
            JobSatisfaction = st.selectbox("Job Satisfaction", options=sorted(train_df['JobSatisfaction'].unique()))
            RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", options=sorted(train_df['RelationshipSatisfaction'].unique()))
            WorkLifeBalance = st.selectbox("Work Life Balance", options=sorted(train_df['WorkLifeBalance'].unique()))

        with st.expander("Experience and Background"):
            TotalWorkingYears = st.number_input("Total Working Years", min_value=0, value=5)
            NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, value=1)
            Education = st.selectbox("Education", options=sorted(train_df['Education'].unique()))
            EducationField = st.selectbox("Education Field", options=sorted(train_df['EducationField'].unique()))

        with st.expander("Performance Metrics"):        
            PerformanceRating = st.selectbox("Performance Rating", options=sorted(train_df['PerformanceRating'].unique()))
            st.session_state["PerformanceRating"] = PerformanceRating
            JobInvolvement = st.selectbox("Job Involvement", options=sorted(train_df['JobInvolvement'].unique()))

        with st.expander("Travel and Commute"):
            DistanceFromHome = st.number_input("Distance From Home (in miles, max 29)", min_value=0, value=10)
            BusinessTravel = st.selectbox("Business Travel", options=sorted(train_df['BusinessTravel'].unique()))
        
        submit_button = st.form_submit_button(label='Predict Turnover')
    
    if submit_button:
        # DataFrame erstellen
        input_data = {
            'Age': Age,
            'DailyRate': DailyRate,
            'DistanceFromHome': DistanceFromHome,
            'HourlyRate': HourlyRate,
            'MonthlyIncome': MonthlyIncome,
            'MonthlyRate': MonthlyRate,
            'NumCompaniesWorked': NumCompaniesWorked,
            'PercentSalaryHike': PercentSalaryHike,
            'TotalWorkingYears': TotalWorkingYears,
            'TrainingTimesLastYear': TrainingTimesLastYear,
            'YearsSinceLastPromotion': YearsSinceLastPromotion,
            'YearsAtCompany': YearsAtCompany,
            'BusinessTravel': BusinessTravel,
            'Department': Department,
            'Education': Education,
            'EducationField': EducationField,
            'Gender': Gender,
            'JobRole': JobRole,
            'MaritalStatus': MaritalStatus,
            'OverTime': OverTime,
            'Over18': Over18,
            'StockOptionLevel': StockOptionLevel,
            'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'JobInvolvement': JobInvolvement,
            'JobSatisfaction': JobSatisfaction,
            'PerformanceRating': PerformanceRating,
            'RelationshipSatisfaction': RelationshipSatisfaction,
            'WorkLifeBalance': WorkLifeBalance,
            'JobLevel': JobLevel
        }
        input_df = pd.DataFrame([input_data])
        return input_df
    else:
        return None

user_input_df = get_user_input()

# ------------------------------
# 7. Data Preprocessing für User Input
# ------------------------------
if user_input_df is not None:
    with st.spinner("Preprocessing user input..."):
        try:
            user_df = user_input_df.drop(drop_columns, axis=1, errors='ignore')  
            user_df[categorical_columns] = user_df[categorical_columns].astype('category')
            
            if user_df.isnull().sum().sum() > 0:
                st.warning("Input data contains missing values. Filling missing values by dropping rows.")
                user_df = user_df.dropna()
            
            for col in columns_to_encode:
                le = label_encoder[col]
                user_df[col] = le.transform(user_df[col])
                user_df[col] = user_df[col].astype('category')
            
            # Finaler Input für prediction vorbereiten 
            X_user = user_df.drop(['YearsAtCompany'], axis=1, errors='ignore')  
            
            # Wieder die feature liste verwenden für die richtige Reihenfolge
            feature_columns = [
                'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
                'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsSinceLastPromotion',
                'JobLevel', 'BusinessTravel', 'Department',
                'Education', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                'OverTime', 'Over18', 'StockOptionLevel', 'EnvironmentSatisfaction',
                'JobInvolvement', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'WorkLifeBalance'
            ]
            
            X_user = X_user[feature_columns]
            
            #Fehlende Features behandeln
            missing_features = set(feature_columns) - set(X_user.columns)
            for feature in missing_features:
                X_user[feature] = 0  
            
            extra_features = set(X_user.columns) - set(feature_columns)
            if extra_features:
                X_user = X_user.drop(columns=extra_features)
            
            required_features = X_train.columns.tolist()
            missing_features_final = set(required_features) - set(X_user.columns)
            if missing_features_final:
                st.error(f"The following required features are missing: {missing_features_final}")
                st.stop()
            
            # Reihenfolge der Variablen an Trainingsdatensatz anpassen
            X_user = X_user[required_features]
            
            st.success("User input preprocessing completed!")
            
        except Exception as e:
            st.error(f"Error during preprocessing user input: {e}")

    # ------------------------------
    # 8. Prediction und Visualisierung
    # ------------------------------
    with st.spinner("Making predictions..."):
        try:
            # Survival Rate predicten 
            survival_functions = rsf_model.predict_survival_function(X_user)
            original_risk_score = rsf_model.predict(X_user)
            hazard_ratio = np.exp(original_risk_score)

            baseline_times = kmf.survival_function_.index.values  # Zeitpunkte der Basis-Überlebensfunktion
            baseline_survival = kmf.survival_function_.values.flatten()

            def compute_conditional_expected_time(hazard_ratio, baseline_survival, baseline_times, t_condition):

                # Finden der Überlebenswahrscheinlichkeit bei t_condition
                idx_condition = np.searchsorted(baseline_times, t_condition, side="right") - 1
                survival_prob_at_t_condition = baseline_survival[idx_condition]

                # Passe die Überlebenswahrscheinlichkeiten der Basisfunktion mithilfe der Hazardratio des Individuums an
                adjusted_survival_probs = (baseline_survival / survival_prob_at_t_condition) ** hazard_ratio
                adjusted_survival_probs[baseline_times <= t_condition] = 0

                # Berechne die bedingte erwartete Überlebenszeit
                time_diffs = np.diff(np.append(0, baseline_times))
                conditional_expected_time = np.sum(adjusted_survival_probs * time_diffs)
                return conditional_expected_time

            # Anwenden der Funktion
            conditional_expected_survival_time = compute_conditional_expected_time(
                hazard_ratio, baseline_survival, baseline_times, user_df['YearsAtCompany'][0]
            )
        
            def compute_conditional_survival_curve_individual(rsf_model, X_individual, t_condition):
 
                survival_curve = rsf_model.predict_survival_function(X_individual)[0]  
            
                time_points = survival_curve.x
                survival_probs = survival_curve.y
            
            
                idx_condition = np.searchsorted(time_points, t_condition, side="right") - 1
                S_t_condition = survival_probs[idx_condition] if idx_condition >= 0 else 1.0
            
                conditional_probs = survival_probs / S_t_condition
                conditional_probs[time_points <= t_condition] = 1.0  # Set survival to 1 for t <= t_condition
                filtered_time_points = time_points[time_points > t_condition]
                filtered_conditional_probs = conditional_probs[time_points > t_condition]
            
                return {"time": filtered_time_points, "survival": filtered_conditional_probs}

            
            conditional_survival_curve = compute_conditional_survival_curve_individual(rsf_model, X_user, user_df['YearsAtCompany'][0])
            
            plt.step(conditional_survival_curve["time"], conditional_survival_curve["survival"], where="post")
            plt.title(f"Conditional Survival Curve for One Individual (t > {user_df['YearsAtCompany'][0]})")
            plt.xlabel("Time")
            plt.ylabel("Conditional Survival Probability")
            plt.grid(True)
            plt.show()
            
            st.title("Conditional Survival Curve")

            # Plot erstellen
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.step(conditional_survival_curve["time"], conditional_survival_curve["survival"], where="post")
            ax.set_title(f"Conditional Survival Curve for One Individual (t > {user_df['YearsAtCompany'][0]})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Conditional Survival Probability")
            ax.grid(True)


        
            # Survival Probility extracten 
            surv_func = list(survival_functions)[0]
            print(surv_func)
            times = conditional_survival_curve["time"]
            survival_prob = conditional_survival_curve["survival"]
            print(f"{survival_prob}")

            selected_times = [4.0, 6.0, 10.0, 20.0, 30.0]
            surv_probs_selected = np.interp(selected_times, times, survival_prob)
            surv_probs_df1 = pd.DataFrame({
                'Time (Years)': selected_times,
                'Survival Probability': surv_probs_selected
            })

            # Plot fig
            fig, ax = plt.subplots(figsize=(10,6))
            ax.step(times, survival_prob, where="post", label="Predicted Survival", linestyle='--', color='blue')
            ax.set_xlabel("Time (Years)", fontsize=12)
            ax.set_ylabel("Survival Probability", fontsize=12)
            ax.set_title("Predicted Survival Function for the Employee", fontsize=14)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=10)

            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.7, label="50% Probability Line")
            ax.annotate("50% Probability", xy=(times[len(times)//2], 0.5), xytext=(5, 0.55),
            arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)
            
            # Plot der allgemeinen Kaplan-Meier Survival Kurve
            kmf = KaplanMeierFitter()
            kmf.fit(durations=train_df_processed['YearsAtCompany'], event_observed=train_df_processed['Attrition'])
            kmf.plot_survival_function(ax=ax, label="Mean Survival Probability", color='orange')
            
            options = np.unique(times.round(2))
            default = [val for val in [4.0, 5.0, 6.0, 10.0, 20.0, 30.0] if val in options]


            # Survival Probabilities an verschiedenen Zeitpunkten
            st.write("### Survival Probabilities at Specific Time Points")
            selected_times = st.multiselect(
                "Select time points (Years) to view survival probabilities:",
                options=options,
                default=default  # Filtered default values
            )

            surv_probs_selected = np.interp(selected_times, surv_func.x, surv_func.y)
            surv_probs_df = pd.DataFrame({
                'Time (Years)': selected_times,
                'Survival Probability': surv_probs_selected
            })

            def generate_insight(prob):
                if prob > 0.7:
                    return "High probability of staying"
                elif prob > 0.6:
                    return "Moderate probability of staying"
                else:
                    return "High risk of leaving"

            surv_probs_df1['Insight'] = surv_probs_df1['Survival Probability'].apply(generate_insight)

            st.write("### Survival Probabilities and Insights")

            # Farben
            def color_survival(val):
                if val > 0.7:
                    return 'background-color: green'
                elif val > 0.4:
                    return 'background-color: yellow'
                else:
                    return 'background-color: lightcoral'


            styled_table = surv_probs_df1.style.applymap(color_survival, subset=['Survival Probability'])

            col1, col2 = st.columns(2)

            with col1:

                st.dataframe(styled_table)

                predicted_year = 3  
                predicted_survival = np.interp(predicted_year, surv_func.x, surv_func.y)
                with st.expander("🔍 Prediction Insight:"):
                    st.write(f"""
                - At year {selected_times[0]}: {surv_probs_selected[0]:.1%} survival probability — most employees stay during the first year.
                - At year {selected_times[-1]}: {surv_probs_selected[-1]:.1%} survival probability — fewer employees remain with the company long-term.
                """)
                with st.expander("Performance Rating:"):
                    if st.session_state.get("PerformanceRating") == 3:
                        st.warning("A salary hike is required to improve the performance rating from 3 to 4!")
        
                        # Salary hike verwendet
                        current_hike = st.session_state.get("PercentSalaryHike", 0)

                        # Threshold für salary hike wurde mit Random Forest berechnet
                        threshold_salary_hike = 19.49768

                        # Berechnung der benötigten salary hike
                        additional_hike_needed = max(0, threshold_salary_hike - current_hike)

                        if additional_hike_needed > 0:
                            st.write(
                                f"An additional salary hike of **{additional_hike_needed:.4f}%** is needed to achieve an outstanding performance rating."
                            )
                        else: 
                            st.success("The performance has reached it's maximum. No additional salary hike is required!")
                    else:
                        st.success("The performance has reached it's maximum. No additional salary hike is required!")



            with col2:
                st.pyplot(fig2)
                st.write(f"")
                with st.expander("Understand what this graph means"):
                    st.write("""
                    Imagine you have a group of employees, and you want to track how long they stay at the company before leaving. The Kaplan-Meier curve shows the probability of employees staying with the company over time.
                    - The curve starts at 100%, because at the beginning, everyone is still at the company.
                    - As time goes on, the curve gradually drops, reflecting that some employees leave.
                    """)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
   
