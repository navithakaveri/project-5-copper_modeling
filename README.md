# project-5-copper_modeling
Industrial copper modeling

**Data Preprocessing:**
The project begins with a data preprocessing step, which involves handling missing values, removing duplicate values, and converting data types to prepare the dataset for analysis and modeling.



**Exploratory Data Analysis (EDA):**
*The EDA process includes both manual and automated approaches.
*Manual EDA involves visualizing key features such as quantity tons, application, thickness, width, and selling price to understand their distributions and relationships.
*Automated EDA is performed using Sweetviz, a library for generating high-density visualizations and comprehensive statistical summaries of datasets.




**Machine Learning (ML) Model Development:**
*The ML component of the project involves building models to predict selling prices and classify leads as WON or LOST.
For predicting selling prices, a machine learning model (possibly a regression model) is trained using features such as quantity tons, application, thickness, width, country, customer, product reference, and item type.


*For classifying leads, a classification model (possibly a decision tree classifier) is trained using features like quantity tons, selling price, application, thickness, width, country, customer, product reference, and item type.
Both models are trained using historical data and saved using pickle for future use.



**Streamlit Web Application:**
*The project includes a Streamlit web application that provides a user-friendly interface for interacting with the data and models.
*Users can navigate through different sections of the application, including About, Data Preprocessing, EDA Process, and Insights, using an option menu.
*In the Data Preprocessing section, users can upload CSV files, preprocess the data, and view summary statistics.
*In the EDA Process section, users can visualize data distributions, correlations, and generate automated EDA reports using Sweetviz.
*In the Insights section, users can use machine learning models to predict selling prices and classify leads based on user inputs.



Overall, the project aims to provide a comprehensive analysis of industrial copper data, empower users to explore and understand the dataset visually, and leverage machine learning models for predictive insights. The Streamlit web application serves as a user-friendly interface to access and interact with different stages of the analysis.






