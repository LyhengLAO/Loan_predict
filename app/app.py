import streamlit as st
import pandas as pd
from interference import predict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model"
test_path = BASE_DIR / "data" / "processed" / "test_processed.csv"

encoding_dict = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "credit_history": {"No Credit History": 0, "Has Credit History": 1}
}

reverse_encoding_dict = {
    feature: {v: k for k, v in mapping.items()}
    for feature, mapping in encoding_dict.items()
}

best_params = pd.read_csv(MODEL_PATH / "best_params.csv", index_col=0)
model_comparison = pd.read_csv(MODEL_PATH / "model_comparison.csv", index_col=0)

test_df = pd.read_csv(test_path)
test_df = test_df.drop(columns=["Loan_ID"])

# ----- init session -----
if "page" not in st.session_state:
    st.session_state.page = "Prédiction"
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "random_data" not in st.session_state:
    st.session_state.random_data = test_df.sample(n=1).iloc[0].tolist()

st.title('Loan Approval Prediction')
st.markdown('This app predicts whether a loan will be approved based on user input features.')

# ===== SIDEBAR =====
st.sidebar.title('Sommaire')
pages = ['détaille sur les modèle','Prédiction', 'Résultat']
page = st.sidebar.selectbox('Choisissez une page', pages, index=pages.index(st.session_state.page))
st.session_state.page = page

# ===== PAGE: détails =====
if page == 'détaille sur les modèle':
    st.subheader('Détail sur les modèles utilisés')
    st.markdown(
        "Nous avons utilisé plusieurs modèles de machine learning pour prédire l'approbation des prêts..."
    )
    st.subheader('Comparaison des modèles')
    st.dataframe(model_comparison)
    st.subheader('Meilleurs hyperparamètres')
    st.dataframe(best_params)

# ===== PAGE: prédiction =====
elif page == 'Prédiction':
    st.subheader('Entrez les détails du demandeur de prêt')
    st.markdown("### Choisis une option :")
    choice = st.selectbox(
        'Choisissez une option',
        ["self input data", "random from test set"]
    )

    st.write(f"Your choice: **{choice}**")
    
    if choice == "self input data":
        st.markdown('### input your data')
        gender = st.selectbox("Gender", list(encoding_dict["Gender"].keys()))
        married = st.selectbox("Married", list(encoding_dict["Married"].keys()))
        dependents = st.selectbox("Dependents", list(encoding_dict["Dependents"].keys()))
        education = st.selectbox("Education", list(encoding_dict["Education"].keys()))
        self_employed = st.selectbox("Self Employed", list(encoding_dict["Self_Employed"].keys()))
        property_area = st.selectbox("Property Area", list(encoding_dict["Property_Area"].keys()))

        applicant_income = st.number_input("Applicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_term = st.number_input("Loan Amount Term (in months)", min_value=0)

        credit_history = st.selectbox("Credit History", list(encoding_dict["credit_history"].keys()))

        data = [
            encoding_dict["Gender"][gender],
            encoding_dict["Married"][married],
            encoding_dict["Dependents"][dependents],
            encoding_dict["Education"][education],
            encoding_dict["Self_Employed"][self_employed],
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            encoding_dict["credit_history"][credit_history],
            encoding_dict["Property_Area"][property_area],
        ]

    elif choice == "random from test set":
    
        st.markdown("### Random test set")

        if st.button("🎲 Randomize again"):
            random_row = test_df.sample(n=1).iloc[0]
            st.session_state.random_data = random_row.values.tolist()

        if st.session_state.random_data is not None:
            data = st.session_state.random_data

            gender = reverse_encoding_dict["Gender"][data[0]]
            married = reverse_encoding_dict["Married"][data[1]]
            dependents = reverse_encoding_dict["Dependents"][data[2]]
            education = reverse_encoding_dict["Education"][data[3]]
            self_employed = reverse_encoding_dict["Self_Employed"][data[4]]
            applicant_income = data[5]
            coapplicant_income = data[6]
            loan_amount = data[7]
            loan_term = data[8]
            credit_history = reverse_encoding_dict["credit_history"][data[9]]
            property_area = reverse_encoding_dict["Property_Area"][data[10]]

            d = pd.DataFrame([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, property_area]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'credit_history', 'Property_Area'])
            st.dataframe(d.transpose(),use_container_width=True, height=423)

    st.subheader('Select Model for Prediction')
    model_selection = st.selectbox("Select Model", model_comparison.index)

    data_values = [gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, property_area]

    if st.button('Predict'):
        model_file = MODEL_PATH / f"model_{model_selection}.joblib"
        result = predict(model_file, data)

        # save for Result page
        st.session_state.result = int(result)
        st.session_state.selected_model = str(model_selection)
        st.session_state.last_data = pd.DataFrame([data_values], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'credit_history', 'Property_Area'])

        # go to Result page
        st.session_state.page = "Résultat"
        st.rerun()

# ===== PAGE: résultat =====
elif page == 'Résultat':
    st.subheader("Prediction Result")

    if st.session_state.result is None:
        st.warning("Aucun résultat pour le moment. Va sur 'Prédiction' et clique Predict.")
    else:
        st.markdown(f"**Modèle choisi :** {st.session_state.selected_model}")

        if st.session_state.result == 1:
            st.success("Loan Approved")
            st.balloons()
        else:
            st.error("Loan Rejected")

        with st.expander("Voir les valeurs encodées envoyées au modèle"):
            st.write(st.session_state.last_data)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Modifier les inputs"):
            st.session_state.page = "Prédiction"
            st.rerun()
    with col2:
        if st.button("🧹 Reset résultat"):
            st.session_state.result = None
            st.session_state.selected_model = None
            st.session_state.last_data = None
            st.rerun()
