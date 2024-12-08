import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ------------------------------
# 1. Konfiguration der Streamlit Seite
# ------------------------------
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("Employee Survival Rate Prediction")
st.write("""
Welcome to the Employee Survival Prediction Page! 
This tool allows you to predict how long an employee will stay at your company based on historical data.
""")

# ------------------------------
# 2. Training Datensatz laden 
# ------------------------------
st.header("📂 Training Dataset")

training_file_path = "DA_Streamlit/Training.xlsx" 

# Training Datensatz war zuerst ein .xls file und das konnte nicht mit openpyxl benutzt werden
# Könnte aber nach Umwandlung verworfen werden
@st.cache_data
def load_training_data(path):
    if not os.path.exists(path):
        st.error(f"Training file '{path}' not found. Please ensure it's in the correct directory.")
        return None
    try:
        file_extension = path.split('.')[-1]
        if file_extension == 'xlsx':
            df = pd.read_excel(path, engine="openpyxl")
        elif file_extension == 'xls':
            df = pd.read_excel(path, engine="xlrd")
        else:
            st.error("Unsupported file format for training data! Please use .xlsx or .xls.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

# Trainingsdaten laden
train_df = load_training_data(training_file_path)

if train_df is not None:
    st.success("Training data loaded successfully!")
    st.write("### Training Dataset Preview")
    st.dataframe(train_df.head())
else:
    st.stop()  

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
    'JobLevel'  # Added JobLevel
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
    st.success("Training data preprocessing completed!")



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
        
        fig_km, ax_km = plt.subplots(figsize=(10,6))
        kmf.plot_survival_function(ax=ax_km)
        ax_km.set_title("Kaplan-Meier Survival Curve")
        ax_km.set_xlabel("Time (Years)")
        ax_km.set_ylabel("Survival Probability")

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
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        DailyRate = st.number_input("Daily Rate", min_value=0, value=500)
        DistanceFromHome = st.number_input("Distance From Home (in miles)", min_value=0, value=10)
        HourlyRate = st.number_input("Hourly Rate", min_value=0, value=50)
        MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
        MonthlyRate = st.number_input("Monthly Rate", min_value=0, value=5000)
        NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, value=1)
        PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, value=10)
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, value=5)
        TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, value=1)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, value=2)
        YearsAtCompany = st.number_input("Years at Company", min_value=0, value=3)
        
        # Kategorischer Input
        BusinessTravel = st.selectbox("Business Travel", options=sorted(train_df['BusinessTravel'].unique()))
        Department = st.selectbox("Department", options=sorted(train_df['Department'].unique()))
        Education = st.selectbox("Education", options=sorted(train_df['Education'].unique()))
        EducationField = st.selectbox("Education Field", options=sorted(train_df['EducationField'].unique()))
        Gender = st.selectbox("Gender", options=sorted(train_df['Gender'].unique()))
        JobRole = st.selectbox("Job Role", options=sorted(train_df['JobRole'].unique()))
        MaritalStatus = st.selectbox("Marital Status", options=sorted(train_df['MaritalStatus'].unique()))
        OverTime = st.selectbox("Over Time", options=sorted(train_df['OverTime'].unique()))
        Over18 = st.selectbox("Over 18", options=sorted(train_df['Over18'].unique()))
        
        StockOptionLevel = st.selectbox("Stock Option Level", options=sorted(train_df['StockOptionLevel'].unique()))
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", options=sorted(train_df['EnvironmentSatisfaction'].unique()))
        JobInvolvement = st.selectbox("Job Involvement", options=sorted(train_df['JobInvolvement'].unique()))
        JobSatisfaction = st.selectbox("Job Satisfaction", options=sorted(train_df['JobSatisfaction'].unique()))
        PerformanceRating = st.selectbox("Performance Rating", options=sorted(train_df['PerformanceRating'].unique()))
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", options=sorted(train_df['RelationshipSatisfaction'].unique()))
        WorkLifeBalance = st.selectbox("Work Life Balance", options=sorted(train_df['WorkLifeBalance'].unique()))
        
        JobLevel = st.number_input("Job Level", min_value=1, max_value=5, value=1)
        
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
            
            # Survival Probility extracten 
            surv_func = list(survival_functions)[0]
            times = surv_func.x
            survival_prob = surv_func.y
            
            # Plot
            fig_user, ax_user = plt.subplots(figsize=(10,6))
            ax_user.step(times, survival_prob, where="post", label="Predicted Survival")
            ax_user.set_xlabel("Time (Years)")
            ax_user.set_ylabel("Survival Probability")
            ax_user.set_title("Predicted Survival Function for the Employee")
            ax_user.set_ylim(0, 1)
            ax_user.legend()
            
            # Plot der allgemeinen Kaplan-Meier Survival Kurve
            kmf = KaplanMeierFitter()
            kmf.fit(durations=train_df_processed['YearsAtCompany'], event_observed=train_df_processed['Attrition'])
            kmf.plot_survival_function(ax=ax_user, label="Overall KM Survival")
        
            
            # Survival Probabilities an verschiedenen Zeitpunkten
            st.write("### Survival Probabilities at Specific Time Points")
            selected_times = st.multiselect(
                "Select time points (Years) to view survival probabilities:",
                options=np.unique(times.round(2)),
                default=[1, 2, 3, 4, 5]
            )
            if selected_times:
                surv_probs_selected = np.interp(selected_times, surv_func.x, surv_func.y)
                
                survival_dict = {time: prob for time, prob in zip(selected_times, surv_probs_selected)}
                surv_probs_df = pd.DataFrame(list(survival_dict.items()), columns=['Time (Years)', 'Survival Probability'])
                st.table(surv_probs_df)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # ------------------------------
   # 9. Variable Manipulation Tool
    # ------------------------------
    st.header("🔧 Variable Manipulation Tool")

    st.write("""
    Use the controls below to adjust the values of key variables and observe how they affect the employee's survival probability.
    This tool helps identify which variables have the most significant impact on improving employee retention.
    """)

    st.subheader("🛠️ Adjust Variables")

    # Initialize session state for manipulated variables
    if 'manipulated_input_done' not in st.session_state:
        st.session_state.manipulated_input_done = False
        st.session_state.manipulated_survival_prob = None
        st.session_state.manipulated_times = None

    with st.expander("Open Variable Manipulation Controls"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Manip_JobLevel = st.slider("Job Level", min_value=1, max_value=5, value=1, help="Adjust the job level (1-5).")
            Manip_OverTime = st.selectbox("Over Time", options=['Yes', 'No'], index=0, help="Adjust the overtime status.")
        
        with col2:
            Manip_EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", options=['Low', 'Medium', 'High', 'Very High'], index=1, help="Adjust the environment satisfaction level.")
            Manip_StockOptionLevel = st.slider("Stock Option Level", min_value=0, max_value=3, value=0, help="Adjust the stock option level.")
        
        with col3:
            Manip_JobSatisfaction = st.selectbox("Job Satisfaction", options=['Low', 'Medium', 'High', 'Very High'], index=1, help="Adjust the job satisfaction level.")
        
        manipulate_button = st.button("Update Survival Probability")

    if manipulate_button:
        with st.spinner("Updating survival probability based on manipulated variables..."):
            try:
                # Create a copy of the original user input data
                manipulated_input = user_input_df.copy()
                
                # Update the manipulated variables in the raw input data
                manipulated_input['JobLevel'] = Manip_JobLevel
                manipulated_input['OverTime'] = Manip_OverTime
                manipulated_input['EnvironmentSatisfaction'] = Manip_EnvironmentSatisfaction
                manipulated_input['StockOptionLevel'] = Manip_StockOptionLevel
                manipulated_input['JobSatisfaction'] = Manip_JobSatisfaction
                
                # Preprocess the manipulated input data
                manipulated_user_df = manipulated_input.drop(drop_columns, axis=1, errors='ignore')
                manipulated_user_df[categorical_columns] = manipulated_user_df[categorical_columns].astype('category')
                
                # Handle missing values if any
                if manipulated_user_df.isnull().sum().sum() > 0:
                    st.warning("Manipulated input data contains missing values. Filling missing values by dropping rows.")
                    manipulated_user_df = manipulated_user_df.dropna()
                
                # Scale numerical columns using the scaler fitted on training data
                
                # Encode categorical variables using the label encoder fitted on training data
                for col in columns_to_encode:
                    le = label_encoder[col]
                    # Handle unseen labels
                    if manipulated_user_df[col].iloc[0] in le.classes_:
                        manipulated_user_df[col] = le.transform(manipulated_user_df[col])
                    else:
                        st.warning(f"Value '{manipulated_user_df[col].iloc[0]}' for '{col}' not seen in training data. Encoding as 0.")
                        manipulated_user_df[col] = 0  # Default encoding
                    manipulated_user_df[col] = manipulated_user_df[col].astype('category')
                
                # Prepare the final input for prediction
                X_manipulated = manipulated_user_df.drop(['YearsAtCompany'], axis=1, errors='ignore')
                
                # Reorder and select only the required features
                X_manipulated = X_manipulated[feature_columns]
                
                # Optional: Handle any missing features by adding them with default values
                missing_features = set(feature_columns) - set(X_manipulated.columns)
                for feature in missing_features:
                    X_manipulated[feature] = 0  # Adjust default values as appropriate
                
                # Identify and remove extra features
                extra_features = set(X_manipulated.columns) - set(feature_columns)
                if extra_features:
                    X_manipulated = X_manipulated.drop(columns=extra_features)
                
                # Verify that all required features are present
                required_features = X_train.columns.tolist()
                missing_features_final = set(required_features) - set(X_manipulated.columns)
                if missing_features_final:
                    st.error(f"The following required features are missing in the manipulated input: {missing_features_final}")
                    st.stop()
                
                # Reorder the columns to match training data
                X_manipulated = X_manipulated[required_features]
                
                # Predict survival function for the manipulated input
                manipulated_survival_functions = rsf_model.predict_survival_function(X_manipulated)
                manipulated_surv_func = list(manipulated_survival_functions)[0]
                manipulated_times = manipulated_surv_func.x
                manipulated_survival_prob = manipulated_surv_func.y
                
                # Store in session state
                st.session_state.manipulated_survival_prob = manipulated_survival_prob
                st.session_state.manipulated_times = manipulated_times
                st.session_state.manipulated_input_done = True
                
                st.success("Survival probability updated based on manipulated variables!")
            except Exception as e:
                st.error(f"Error during variable manipulation: {e}")

    # ------------------------------
    # 10. Display Manipulated Survival Function and Recommendations
    # ------------------------------

    if st.session_state.get('manipulated_input_done', False):
        with st.spinner("Displaying manipulated survival probability..."):
            try:
                manipulated_surv_prob = st.session_state.manipulated_survival_prob
                manipulated_times = st.session_state.manipulated_times
                
                # Plot the manipulated survival function
                fig_manip, ax_manip = plt.subplots(figsize=(10,6))
                ax_manip.step(manipulated_times, manipulated_surv_prob, where="post", label="Manipulated Survival")
                ax_manip.set_xlabel("Time (Years)")
                ax_manip.set_ylabel("Survival Probability")
                ax_manip.set_title("Survival Function After Variable Manipulation")
                ax_manip.set_ylim(0, 1)
                ax_manip.legend()
                
                # Plot the original survival function for comparison
                # Retrieve original survival function from user input
                survival_functions = rsf_model.predict_survival_function(X_user)
                surv_func = list(survival_functions)[0]
                times = surv_func.x
                survival_prob = surv_func.y
                ax_manip.step(times, survival_prob, where="post", label="Original Survival", linestyle='--')
                ax_manip.legend()
                
                st.pyplot(fig_manip)
                
                # Display survival probabilities at specific time points for manipulated input
                st.write("### Survival Probabilities at Specific Time Points (Manipulated)")
                manipulated_selected_times = st.multiselect(
                    "Select time points (Years) to view manipulated survival probabilities:",
                    options=np.unique(manipulated_times.round(2)),
                    default=[1, 2, 3, 4, 5]
                )
                if manipulated_selected_times:
                    # Using NumPy's interp for interpolation
                    manipulated_surv_probs_selected = np.interp(manipulated_selected_times, manipulated_times, manipulated_surv_prob)
                    
                    survival_dict_manip = {time: prob for time, prob in zip(manipulated_selected_times, manipulated_surv_probs_selected)}
                    surv_probs_df_manip = pd.DataFrame(list(survival_dict_manip.items()), columns=['Time (Years)', 'Survival Probability'])
                    st.table(surv_probs_df_manip)
                
                # ------------------------------
                # 11. Feature Importances and Recommendations
                # ------------------------------
                
                st.subheader("📊 Feature Importances")
                feature_importances = pd.Series(rsf_model.feature_importances_, index=X_train.columns)
                feature_importances = feature_importances.sort_values(ascending=False)
                fig_imp_manip, ax_imp_manip = plt.subplots(figsize=(10,6))
                feature_importances.plot(kind='bar', ax=ax_imp_manip)
                ax_imp_manip.set_title("Feature Importances")
                ax_imp_manip.set_ylabel("Importance Score")
                ax_imp_manip.set_xlabel("Features")
                st.pyplot(fig_imp_manip)
                
                # Recommendations based on top features
                top_features = feature_importances.head(5).index.tolist()
                st.subheader("💡 Recommendations")
                st.write("""
                Based on the feature importances, the following variables have the most significant impact on improving employee retention:
                """)
                for i, feature in enumerate(top_features, 1):
                    st.markdown(f"**{i}. {feature}**")
                
                st.write("""
                **Recommendations:**
                - **Focus on improving these areas to enhance employee retention.**
                - **Consider implementing training programs, offering competitive stock options, and addressing job satisfaction concerns.**
                """)
                
            except Exception as e:
                st.error(f"Error displaying manipulated survival probability: {e}")

   