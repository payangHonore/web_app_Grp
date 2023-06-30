import streamlit as st 
import pandas as pd
import datetime
import pandas_profiling


from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class

url ="https://www.linkedin.com/in/payang-honor%C3%A9-3a2908246"

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def main():
    st.title("Porjet Machine Learning sur le Cars price Predictionn")
    st.sidebar.write("[Auteurs: Dev-Dat Groupe 5](%s)" % url)
    st.sidebar.markdown(
        "** This web app is No-Code tool for Exploratory Data Analysis and Building Machine Learning Model for Regression and Classification"
        "1. Load your dataset file (CSV file);\n"
        "2. Click on *Profile Dataset* button in order to generate the pandas profiling of the dataset;\n"
        "3. Choose your target column;\n"
        "4. Choose the Machine learning Task( Regression or Classification);\n"
        "5. Click on *Run Modelling* in order to start the training process."
        "When the model is built, you can view the serults like the pipeline model, Residuals plot, ROC curve, confusion Matrix;\n"
        "\n6. Download the pipeline model in your local computer"

    )

    file = st.file_uploader("upload your dataset in csv format", type=["csv"])

    if file is not None:
        data = load_data(file)
        st.dataframe(data.head())

        profile = st.button("profile dataset")
        if profile:
            profile_df = pandas_profiling.ProfileReport(data)
           # profile_df = data.profile_report()
            st_profile_report(profile_df)

        target = st.selectbox("select the target variable",data.columns)
        task = st.selectbox("select a ML task", ["Regression","Classification"])

        if task == "Regression":
            if st.button("Run Modelling"):
                exo_reg = setup_reg(data, target=target)
                model_reg = compare_models_reg()
                #save_model_class(model_class, "best_class_model")
                save_model_reg(model_reg, "best_reg_model")
                st.success("Regression Model build successfully")

                #Resultat
                st.write("Residuals")
                plot_model_reg(model_reg, plot ='residuals', save=True)
                st.image("Residuals.png")

                st.write("Feature Importance")
                plot_model_reg(model_reg, plot ='feature', save=True)
                st.image("Feature Importance.png")

                with open('best_reg_model.pkl', 'rb') as f:
                    st.download_button('Download Pipeline Model', f, file_name="best_reg_model.pkl")

        if task == "Classification":
            if st.button("Run Modelling"):
                exp_class = setup_reg(data, target = target)
                model_class = compare_models_class()
                save_model_class(model_class, "best_class_model")
                st.success("Classification Model build successfully")

                #Resultat
                col5, col6 = st.columns(2)
                with col5:
                    st.write("ROC curve")
                    plot_model_class(model_class, save=True)
                    st.image("AUC.png")

                with col6:
                    st.write("Classification Report")
                    plot_model_class(model_class, plot='class_report', save=True)
                    st.image("class_report.png")

                col7, col8 = st.columns(2)
                with col7:
                    st.write("Confusion Matrix")
                    plot_model_class(model_class, plot='confusion_matrix', save=True)
                    st.image("Confusion Matrix.png")

                with col8:
                    st.write("Feature Importance ")
                    plot_model_class(model_class, plot='feature', save=True)
                    st.image("Feature Importanceç.png")

                # Download the Model
                with open('best_class_model.pkl', 'rb') as f:
                    st.download_button('Download Model', f, file_name="best_class_model.pkl")

    else:
        st.image("https://github.com/payangHonore/DB_globalService/blob/main/image%20ML%20Modif.jpeg?raw=true")

                


if __name__=='__main__':
    main()
    