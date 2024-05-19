import pandas as pd
import numpy as np
import sklearn

import streamlit as st
from streamlit_option_menu import option_menu
from matplotlib import pyplot as plt
import seaborn as s
from PIL import Image
import sklearn
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sklearn
print(sklearn.__version__)  # Should print 0.24.2 or the version you specified


import sweetviz as sv
import io
import re
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV





st.set_page_config(page_title='INDUSTRIAL COPPER',layout="wide")
data=pd.read_csv(r"C:\Users\navit\Downloads\coper.csv")
df=pd.DataFrame(data)


# Load the dataset
def about():
    col1,col2=st.columns([4,4])
    with col1:
        print(sklearn.__version__) 
        st.title("INDUSTRIAL COPPER MODELING")
        icon=Image.open(r"C:\Users\navit\OneDrive\Pictures\IMAGES\coper.png")
        st.image(icon,use_column_width=True)
    with col2:
        st.title("DESCRIPTION")
        st.write('''The Industrial Copper Modeling project aims to address challenges in the copper industry related to sales prediction and lead classification. It involves the development of machine learning models for predicting 
                 selling prices and classifying leads as either successful (WON) or unsuccessful (LOST).''')
        st.title("STEPS involved in the analysis")
        st.header("PYTHON SCRIPTING")
        st.header("DATA PREPROCESSING")
        st.write("Handled missing values")
        st.write("Remove duplicates values")
        st.write("Data type conversion")
        st.header("EDA PROCESS")
        st.write("Manual EDA process")
        st.write("Automated EDA process with sweetviz")
        st.header("STREAMLIT")
        st.write("created user friendly web application easy access and understanding of user")

            
def datapre(df1):
    #type conversion
    df1['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
    df1['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
    df1['customer'] = pd.to_numeric(df['customer'], errors='coerce')
    df1['country'] = pd.to_numeric(df['country'], errors='coerce')
    df1['application'] = pd.to_numeric(df['application'], errors='coerce')
    df1['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
    df1['width'] = pd.to_numeric(df['width'], errors='coerce')
    df1['material_ref'] = df['material_ref'].str.lstrip('0')
    df1['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
    df1['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
    df1['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
    df1.material_ref.fillna('unknown',inplace=True)
    df2=df1.dropna()
    return df2
def show_shape():
    st.write(df.shape) 
def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s
def show_values(df3):
    missing_values_count = df3.isnull().sum()
    st.table(missing_values_count)

        
def eda(df1):
    tab1,tab2=st.tabs(['EDA PROCESS','AUTO EDA PROCESS WITH SWEETVIZ'])
    with tab1:
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title("QUANTITY TONS")
        s.distplot(df['quantity tons'])
        st.pyplot() 
        
        st.title("APPLICATION")
        s.distplot(df['application'])
        st.pyplot() 
        
        st.title("THICKNESS")
        s.distplot(df['thickness'])
        st.pyplot() 
        
        st.title("WIDTH")
        s.distplot(df['width'])
        st.pyplot() 
        
        st.title("selling_price")
        s.distplot(df['selling_price'])
        st.pyplot() 
        
        plt.figure(figsize=(16,8))
        s.boxplot(data=df, y="status", x="selling_price")
        plt.title(' ')
        st.pyplot()

    # Identify and handle problematic columns
        fig = plt.figure(figsize=(25,25))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        x=df[['quantity tons','application','thickness','width','selling_price','country','customer','product_ref']].corr()

        heatmap=s.heatmap(x, annot=True, cmap="YlGnBu")
        st.pyplot(heatmap.figure)
       
        
    with tab2:
        st.title("Auto EDA process")
      # Perform automated EDA using Sweetviz
        report = sv.analyze(df)
      # Save the Sweetviz report as an HTML file
        report_file_path = "sweetviz_report.html"
        report.show_html(report_file_path)
        # Read the HTML file
        with open(report_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Display the HTML content within the Streamlit app
        st.components.v1.html(html_content, height=1000, scrolling=True)
        
   
def ml():
    
   
    tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"]) 
    with tab1:   
        status_options=data['status'].unique()  
        item_options=data['item type'].unique()
        country_options=data['country'].unique()
        application_options =data['application'].unique()
        product =data['product_ref'].unique() 
        
     # Define the widgets for user input
        with st.form("my_form"):
            col1,col2=st.columns([5,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col2:               
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
            
            import pickle
            
            with open(r"model.pkl", 'rb') as file:
                
                loaded_model = pickle.load(file)
            with open(r"scaler.pkl", 'rb') as f:
                
                scaler_loaded = pickle.load(f)

            with open(r"t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)
                
            
                
         

            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be), axis=1)
            st.write(new_sample)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
        
   

            

    

    with tab2: 
    
        with st.form("my_form1"):
            col1,col2=st.columns([5,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col2:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
            import pickle
            with open(r"clsmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r"cscaler.pkl", 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)
    

           
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
           
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')

                
st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Navitha kaveri</h6>', unsafe_allow_html=True )  

def main():
    
    page=option_menu(" ",["ABOUT","DATA PREPROCESSING","EDA PROCESS","INSIGHTS"],orientation='horizontal')
    if page=="ABOUT":
        about()
    elif page=="DATA PREPROCESSING":
        
        st.title('Data Preprocessing App')
    
    # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df_ = pd.read_csv(uploaded_file)
            
            st.write("Preview of the uploaded file:")
            st.dataframe(df_.head())
           
            # Data preprocessing
            df_processed = datapre(df_)
            
            
            # dataframe shape
            st.subheader("Dataframe Shape")
            show_shape()
            
            #  dataframe information
            st.subheader("Dataframe Information")
            st.text(show_info(df_processed))
            
            # missing values
            st.subheader("Missing Values")
            show_values(df_processed)
        
    elif page=="EDA PROCESS":
        eda(df)

    elif page=="INSIGHTS":
        ml()

if __name__ == "__main__":
    main()