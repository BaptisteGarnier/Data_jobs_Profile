# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:20:41 2022

@authors: baptisteg, benoitm, ericg
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
sns.set_theme()
import streamlit as st

#import pydeck as pdk
#import altair as alt
#import matplotlib.pylab as plt

st.set_page_config(layout="wide")

# Création des pages du rapport Streamlit PYGRAAL
#*************************************************
titre = st.container()
intro = st.container()
dataset = st.container()
preprocessing = st.container()
MLalgorithm = st.container()
demo = st.container()
conclusion = st.container()
#*************************************************


# Mise en page du MENU du rapport
#*********************************
st.sidebar.title("Py Graal - Data Jobs")
st.sidebar.header("Sommaire")
option = st.sidebar.radio('',('Introduction', 'Présentation du jeu de données','Prétraitement des données','Stratégie et Modélisation','Démo','Conclusion'))
#*********************************


# import des fichiers de données utilisés pour l'étude
#******************************************************
@st.cache()
def load_df(dataframe):
    df = pd.read_csv(dataframe)
    return df

df = load_df('kaggle_survey_2020_responses.csv')
lignes = df.shape[0]
colonnes = df.shape[1]

df= df.drop('Time from Start to Finish (seconds)',axis=1)
df = df.drop_duplicates(keep='first')
df = df.drop([0],axis=0)


@st.cache()
def load_df_pro(dataframe):
    df_pro = pd.read_csv(dataframe)
    return df_pro

df_pro = load_df_pro('df_pro.csv')

@st.cache()
def load_df_clean(dataframe):
    df_clean = pd.read_csv(dataframe)
    return df_clean

df_clean = load_df_clean('df_clean.csv')

table_quest = pd.read_csv('table_quest.csv', index_col=0)
#******************************************************


# Création d'une liste exhaustive de pays
#*****************************************
@st.cache()
def load_df_pro(dataframe):
    pays = pd.read_csv(dataframe)
    return pays
pays = load_df_pro('Country_List.csv')

Countries = []

for i in pays.Name:
    Countries.append(i)
#*****************************************


#**********************************
# Définition de liste de questions
#**********************************

Q1 = "Q1 - What is your age (# years)?"
Q2 = "Q2 - What is your gender?"
Q3 = "Q3 - In which country do you currently reside?"
Q4 = "Q4 - What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"
Q5 = "Q5 - Select the title most similar to your current role (or most recent title if retired):"
Q6 = "Q6 - For how many years have you been writing code and/or programming?"
Q7 = "Q7 - What programming languages do you use on a regular basis? (Select all that apply)"
Q8 = "Q8 - What programming language would you recommend an aspiring data scientist to learn first?"
Q9 = "Q9 - Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply)"
        
Q10 = "Q10 - Which types of specialized hardware do you use on a regular basis? (Select all that apply)"
Q11 = "Q11 - What type of computing platform do you use most often for your data science projects?"
Q12 = "Q12 - Which types of specialized hardware do you use on a regular basis? (Select all that apply)"
Q13 = "Q13 - Approximately how many times have you used a TPU (tensor processing unit)?"
Q14 = "Q14 - What data visualization libraries or tools do you use on a regular basis? (Select all that apply)"
Q15 = "Q15 - For how many years have you used machine learning methods?"
Q16 = "Q16 - Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply)"
Q17 = "Q17 - Which of the following ML algorithms do you use on a regular basis? (Select all that apply)"
Q18 = "Q18 - Which categories of computer vision methods do you use on a regular basis? (Select all that apply)"
Q19 = "Q19 - Which of the following natural language processing (NLP) methods do you use on a regular basis? (Select all that apply)"
        
Q20 = "Q20 - What is the size of the company where you are employed?"
Q21 = "Q21 - Approximately how many individuals are responsible for data science workloads at your place of business?"
Q22 = "Q22 - Does your current employer incorporate machine learning methods into their business"
Q23 = "Q23 - Select any activities that make up an important part of your role at work: (Select all that apply)"
Q24 = "Q24 - What is your current yearly compensation (approximate $USD)?"
Q25 = "Q25 - Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?"
Q26A = "Q26 - Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply)"
Q26B = "Q26 - Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply)."
Q27A = "Q27 - Do you use any of the following cloud computing products on a regular basis? (Select all that apply)"
Q27B = "Q27 - Do you use any of the following cloud computing products on a regular basis? (Select all that apply)."
Q28A = "Q28 - Do you use any of the following machine learning products on a regular basis? (Select all that apply)"
Q28B = "Q28 - Do you use any of the following machine learning products on a regular basis? (Select all that apply)."
Q29A = "Q29 - Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply)"
Q29B = "Q29 - Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply)."        
        
Q30 = "Q30 - Which of the following big data products (relational database, data warehouse, data lake, or similar) do you use most often?"
Q31A = "Q31 - Which of the following business intelligence tools do you use on a regular basis? (Select all that apply)"
Q31B = "Q31 - Which of the following business intelligence tools do you use on a regular basis? (Select all that apply)."
Q32 = "Q32 - Which of the following business intelligence tools do you use most often?"
Q33A = "Q33 - Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis? (Select all that apply"
Q33B = "Q33 - Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis? (Select all that apply."
Q34A = "Q34 - Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis? (Select all that apply)"
Q34B = "Q34 - Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis? (Select all that apply)."
Q35A = "Q35 - Do you use any tools to help manage machine learning experiments? (Select all that apply)"
Q35B = "Q35 - Do you use any tools to help manage machine learning experiments? (Select all that apply)."
Q36 = "Q36 - Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply)"
Q37 = "Q37 - On which platforms have you begun or completed data science courses? (Select all that apply)"
Q38 = "Q38 - What is the primary tool that you use at work or school to analyze data? (Include text response)"
Q39 = "Q39 - Who/what are your favorite media sources that report on data science topics? (Select all that apply)"

#**********************************



#********************************************************       
# Définition de liste des choix de réponse des questions
#********************************************************

OptionsQ1 = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+'] 
OptionsQ2 = ['Man','Woman','Nonbinary','Prefer not to say','Prefer to self-describe'] 
OptionsQ4 = ['No formal education past high school',
             'Some college/university study without earning a bachelor’s degree',
             'Bachelor’s degree','Master’s degree',' Doctoral degree',
             'Professional degree','I prefer not to answer']    
OptionsQ5 = ['Business Analyst','Data Analyst','Data Engineer','Data Scientist',
                'DBA/Database Engineer','Machine Learning Engineer','Product/Project Manager',
                'Research Scientist','Software Engineer','Statistician',
                'Student','Currently not employed','Other']
OptionsQ6 = ['I have never written code','< 1 years','1-2 years',
            '3-5 years','5-10 years','10-20 years','20+ years'] 
OptionsQ7 = ['Python','R','SQL','C','C++','Java','Javascript',
                'Julia','Swift','Bash','MATLAB','None','Other']           
OptionsQ8 = ['Python','R','SQL','C','C++','Java','Javascript',
             'Julia','Swift','Bash','MATLAB','None','Other']
OptionsQ9 = ['JupyterLab (or products based off of Jupyter)','RStudio','Visual Studio',
                'Visual Studio Code (VSCode)','PyCharm','Spyder','Notepad++',
                'Sublime Text','Vim, Emacs, or similar','MATLAB','None','Other']
OptionsQ10 = ['Kaggle Notebooks','Colab Notebooks','Azure Notebooks','Paperspace / Gradient',
                'Binder / JupyterHub','Code Ocean','IBM Watson Studio','Amazon Sagemaker Studio',
                'Amazon EMR Notebooks','Google Cloud AI Platform Notebooks',
                'Google Cloud Datalab Notebooks','Databricks Collaborative Notebooks','None','Other']
OptionsQ11 = ['A personal computer or laptop','A deep learning workstation (NVIDIA GTX, LambdaLabs, etc)',
                     'A cloud computing platform (AWS, Azure, GCP, hosted notebooks, etc)','None','Other']
OptionsQ12 = ['GPUs','TPUs','None','Other']
OptionsQ13 = ['Never','Once','2-5 times','6-25 times','More than 25 times']
OptionsQ14 = ['Matplotlib','Seaborn','Plotly / Plotly Express','Ggplot / ggplot2','Shiny',
                'D3 js','Altair','Bokeh','Geoplotlib','Leaflet / Folium','None','Other']
OptionsQ15 = ['I do not use machine learning methods','Under 1 year','1-2 years','2-3 years',
              '3-4 years','4-5 years','5-10 years','10-20 years','20 or more years']
OptionsQ16 = ['Scikit-learn','TensorFlow','Keras','PyTorch','Fast.ai','MXNet','Xgboost','LightGBM',
                'CatBoost','Prophet','H2O 3','Caret','Tidymodels','JAX','None','Other']
OptionsQ17 = ['Linear or Logistic Regression','Decision Trees or Random Forests',
                'Gradient Boosting Machines (xgboost, lightgbm, etc)','Bayesian Approaches',
                'Evolutionary Approaches','Dense Neural Networks (MLPs, etc)',
                'Convolutional Neural Networks','Generative Adversarial Networks',
                'Recurrent Neural Networks','Transformer Networks (BERT, gpt-3, etc)','None','Other']
OptionsQ18 = ['General purpose image/video tools (PIL, cv2, skimage, etc)',
                'Image segmentation methods (U-Net, Mask R-CNN, etc)',
                'Object detection methods (YOLOv3, RetinaNet, etc)',
                'Image classification and other general purpose networks (VGG, Inception, ResNet,ResNeXt, NASNet, EfficientNet, etc)',
                'Generative Networks (GAN, VAE, etc)','None','Other']
OptionsQ19 = ['Word embeddings/vectors (GLoVe, fastText, word2vec)','Encoder-decoder models (seq2seq, vanilla transformers)',
                'Contextualized embeddings (ELMo, CoVe)','Transformer language models (GPT-3, BERT, XLnet, etc)','None','Other']
OptionsQ20 = ['0-49 employees','50-249 employees','250-999 employees',
              '1000-9,999 employees','10,000 or more employees']
OptionsQ21 = ['0','1-2','3-4','5-9','10-14','15-19','20+']
OptionsQ22 = ['We are exploring ML methods (and may one day put a model into production)',
                'We use ML methods for generating insights (but do not put working models into production)',
                'We recently started using ML methods (i.e., models in production for less than 2 years)',
                'We have well established ML methods (i.e., models in production for more than 2 years)',
                'No (we do not use ML methods)','I do not know']
OptionsQ23 = ['Analyze and understand data to influence product or business decisions',
                'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
                'Build prototypes to explore applying machine learning to new areas',
                'Build and/or run a machine learning service that operationally improves my product or workflows',
                'Experimentation and iteration to improve existing ML models',
                'Do research that advances the state of the art of machine learning',
                'None of these activities are an important part of my role at work','Other']
OptionsQ25 = ['$0 ($USD)','$1-$99','$100-$999','$1000-$9,999','$10,000-$99,999','$100,000 or more ($USD)']
OptionsQ26 = ['Amazon Web Services (AWS)','Microsoft Azure','Google Cloud Platform (GCP)',
              'IBM Cloud / Red Hat','Oracle Cloud','SAP Cloud','Salesforce Cloud',
              'VMware Cloud','Alibaba Cloud','Tencent Cloud','None','Other']
OptionsQ27 = ['Amazon EC2','AWS Lambda','Amazon Elastic Container Service','Azure Cloud Services',
                'Microsoft Azure Container Instances','Azure Functions','Google Cloud Compute Engine',
                'Google Cloud Functions','Google Cloud Run','Google Cloud App Engine','No / None','Other']
OptionsQ28 = ['Amazon SageMaker','Amazon Forecast','Amazon Rekognition',
              'Azure Machine Learning Studio','Azure Cognitive Services',
              'Google Cloud AI Platform / Google Cloud ML Engine','Google Cloud Video AI',
              'Google Cloud Natural Language','Google Cloud Vision AI','No / None','Other']
OptionsQ29 = ['MySQL','PostgreSQL','SQLite','Oracle Database','MongoDB','Snowflake','IBM Db2',
                'Microsoft SQL Server','Microsoft Access','Microsoft Azure Data Lake Storage',
                'Amazon Redshift','Amazon Athena','Amazon DynamoDB','Google Cloud BigQuery',
                'Google Cloud SQL','Google Cloud Firestore','None','Other']
OptionsQ30 = ['MySQL','PostgreSQL','SQLite','Oracle Database','MongoDB','Snowflake','IBM Db2',
                'Microsoft SQL Server','Microsoft Access','Microsoft Azure Data Lake Storage',
                'Amazon Redshift','Amazon Athena','Amazon DynamoDB','Google Cloud BigQuery',
                'Google Cloud SQL','Google Cloud Firestore','None','Other']
OptionsQ31 = ['Amazon QuickSight','Microsoft Power BI','Google Data Studio','Looker',
                'Tableau','Salesforce','Einstein Analytics','Qlik','Domo','TIBCO Spotfire',
                'Alteryx','Sisense','SAP Analytics Cloud','None','Other']
OptionsQ32 = ['Amazon QuickSight','Microsoft Power BI','Google Data Studio','Looker','Tableau',
                'Salesforce','Einstein Analytics','Qlik','Domo','TIBCO Spotfire','Alteryx','Sisense',
                'SAP Analytics Cloud','None','Other']
OptionsQ33 = ['Automated data augmentation (e.g. imgaug, albumentations)',
                'Automated feature engineering/selection (e.g. tpot, boruta_py)',
                'Automated model selection (e.g. auto-sklearn, xcessiv)',
                'Automated model architecture searches (e.g. darts, enas)',
                'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)',
                'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)',
                'No / None','Other']
OptionsQ34 = ['Google Cloud AutoML','H20 Driverless AI','Databricks AutoML','DataRobot AutoML',
                'Tpot','Auto-Keras','Auto-Sklearn','Auto_ml','Xcessiv','MLbox','No / None','Other']
OptionsQ35 = ['Neptune.ai','Weights & Biases','Comet.ml','Sacred + Omniboard','TensorBoard',
                'Guild.ai','Polyaxon','Trains','Domino Model Monitor','No / None','Other']
OptionsQ36 = ['Plotly Dash','Streamlit','NBViewer','GitHub','Personal blog','Kaggle',
                'Colab','Shiny','None / I do not share my work publicly','Other']
OptionsQ37 = ['Coursera','edX','Kaggle Learn Courses','DataCamp','Fast.ai','Udacity','Udemy',
                'LinkedIn Learning','Cloud-certification programs (direct from AWS, Azure, GCP, or similar)',
                'University Courses (resulting in a university degree)','None','Other']
OptionsQ38 = ['Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
                'Advanced statistical software (SPSS, SAS, etc.)',
                'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
                'Local development environments (RStudio, JupyterLab, etc.)',
                'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)','Other']
OptionsQ39 = ['Twitter (data science influencers)',"Email newsletters (Data Elixir, O'Reilly Data & AI, etc)",
                'Reddit (r/machinelearning, etc)','Kaggle (notebooks, forums, etc)',
                'Course Forums (forums.fast.ai, Coursera forums, etc)','YouTube (Kaggle YouTube, Cloud AI Adventures, etc)',
                'Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)',
                'Blogs (Towards Data Science, Analytics Vidhya, etc)',
                'Journal Publications (peer-reviewed journals, conference proceedings, etc)',
                'Slack Communities (ods.ai, kagglenoobs, etc)','None','Other']

#********************************************************


# Définition de liste des annotations des questions
#***************************************************      
NotesQ18 = "Question 18 (which specific ML methods) was only asked to respondents that selected the relevant answer choices for Question 17 (which categories of algorithms)."
NotesQ19 = "Question 19 (which specific ML methods) was only asked to respondents that selected the relevant answer choices for Question 17 (which categories of algorithms)."
NotesQ27 = "Question 27 (which specific AWS/Azure/GCP products) was only asked to respondents that selected the relevant answer choices for Question 26-A (which of the following companies)."
NotesQ28 = "Question 28 (which specific AWS/Azure/GCP products) was only asked to respondents that selected the relevant answer choices for Question 26-A (which of the following companies)."
NotesQ30 = "Question 30 (which specific product) was only asked to respondents that selected more than one choice for Question 29-A (which of the following products)."
NotesQ32 = "Question 32 (which specific product) was only asked to respondents that selected more than one choice for Question 31-A (which of the following products)."
NotesQ34 = "Question 34 (which specific product) was only asked to respondents that answered affirmatively to Question 33-A (which of the following categories of products)."
Notes_PartB = "Non-professionals received questions with an alternate phrasing (questions for non-professionals asked what tools they hope to become familiar with in the next 2 years instead of asking what tools they use on a regular basis). Non-professionals were defined as students, unemployed, and respondents that have never spent any money in the cloud."
#***************************************************  


#*****************************************************
# Création des dictionnaires relatifs aux questions :
# intitulés, choix, annotations
#*****************************************************

dict_quest = {Q1:OptionsQ1,Q2:OptionsQ2,Q4:OptionsQ4,Q5:OptionsQ5,Q6:OptionsQ6,Q7:OptionsQ7,Q8:OptionsQ8,Q9:OptionsQ9,
              Q10:OptionsQ10,Q11:OptionsQ11,Q12:OptionsQ12,Q13:OptionsQ13,Q14:OptionsQ14,Q15:OptionsQ15,Q16:OptionsQ16,Q17:OptionsQ17,Q18:OptionsQ18,Q19:OptionsQ19,      
              Q20:OptionsQ20,Q21:OptionsQ21,Q22:OptionsQ22,Q23:OptionsQ23,Q25:OptionsQ25,Q26A:OptionsQ26,Q27A:OptionsQ27,Q28A:OptionsQ28,Q29A:OptionsQ29,
              Q30:OptionsQ30,Q31A:OptionsQ31,Q32:OptionsQ32,Q33A:OptionsQ33,Q34A:OptionsQ34,Q35A:OptionsQ35,Q36:OptionsQ36,Q37:OptionsQ37,Q38:OptionsQ38,Q39:OptionsQ39}

dict_notes = {Q18:NotesQ18,Q19:NotesQ19,Q27A:NotesQ27,Q28A:NotesQ28,Q30:NotesQ30,Q32:NotesQ32,Q34A:NotesQ34}
              
dict_notes2 = {Q26A:Notes_PartB,Q27A:Notes_PartB,Q28A:Notes_PartB,Q29A:Notes_PartB,
              Q31A:Notes_PartB,Q33A:Notes_PartB,Q34A:Notes_PartB,Q35A:Notes_PartB}

#*****************************************************


#************************************************************************
# Création de sub dataframes propres à chaque question à choix multiples
#************************************************************************

dfQ7=df.iloc[:,6:19]       #Q7
dfQ9=df.iloc[:,20:32]      #Q9
dfQ10=df.iloc[:,32:46]     #Q10
dfQ12=df.iloc[:,47:51]     #Q12
dfQ14=df.iloc[:,52:64]     #Q14
dfQ16=df.iloc[:,65:81]     #Q16
dfQ17=df.iloc[:,81:93]     #Q17
dfQ18=df.iloc[:,93:100]    #Q18
dfQ19=df.iloc[:,100:106]   #Q19
dfQ23=df.iloc[:,109:117]   #Q23
dfQ26A=df.iloc[:,119:131]  #Q26A
dfQ27A=df.iloc[:,131:143]  #Q27A
dfQ28A=df.iloc[:,143:154]  #Q28A
dfQ29A=df.iloc[:,155:172]  #Q29A
dfQ31A=df.iloc[:,173:188]  #Q31A
dfQ33A=df.iloc[:,189:197]  #Q33A
dfQ34A=df.iloc[:,197:209]  #Q34A
dfQ35A=df.iloc[:,209:220]  #Q35A
dfQ36=df.iloc[:,220:230]   #Q36
dfQ37=df.iloc[:,230:242]   #Q37
dfQ39=df.iloc[:,243:255]   #Q39
dfQ26B=df.iloc[:,255:267]  #Q26B
dfQ27B=df.iloc[:,267:279]  #Q27B
dfQ28B=df.iloc[:,279:290]  #Q28B
dfQ29B=df.iloc[:,290:308]  #Q29B
dfQ31B=df.iloc[:,308:323]  #Q31B
dfQ33B=df.iloc[:,323:331]  #Q33B
dfQ34B=df.iloc[:,331:343]  #Q34B
dfQ35B=df.iloc[:,343:354]  #Q35B

#************************************************************************


#****************************************************
# Créations de Listes et de Dictionnaires facilitant 
# le Preprocessing et la Datavisualization
#****************************************************

LabelsdfQ7 = []
LabelsdfQ9 = []
LabelsdfQ10 = []
LabelsdfQ12 = []
LabelsdfQ14 = []
LabelsdfQ16 = []
LabelsdfQ17 = []
LabelsdfQ18 = []
LabelsdfQ19 = []
LabelsdfQ23 = []
LabelsdfQ26A = []
LabelsdfQ27A = []
LabelsdfQ28A = []
LabelsdfQ29A = []
LabelsdfQ31A = []
LabelsdfQ33A = []
LabelsdfQ34A = []
LabelsdfQ35A = []
LabelsdfQ36 = []
LabelsdfQ37 = []
LabelsdfQ39 = []
LabelsdfQ26B = []
LabelsdfQ27B = []
LabelsdfQ28B = []
LabelsdfQ29B = []
LabelsdfQ31B = []
LabelsdfQ33B = []
LabelsdfQ34B = []
LabelsdfQ35B = []


ListeQL = [Q7,Q9,Q10,Q12,Q14,
            Q16,Q17,Q18,Q19,Q23,
            Q26A,Q27A,Q28A,Q29A,
            Q31A,Q33A,Q34A,Q35A,
            Q36,Q37,Q39,
            Q26B,Q27B,Q28B,Q29B,
            Q31B,Q33B,Q34B,Q35B]

ListeQU = [Q1,Q2,Q3,Q4,Q5,Q6,Q8,
           Q11,Q13,Q15,Q20,Q21,Q22,
           Q24,Q25,Q30,Q32,Q38]

ListeQM = [Q7,Q9,Q10,Q12,Q14,
            Q16,Q17,Q18,Q19,Q23,
            Q36,Q37,Q39]

ListeQMab = [Q26A,Q27A,Q28A,Q29A,
            Q31A,Q33A,Q34A,Q35A,
            Q26B,Q27B,Q28B,Q29B,
            Q31B,Q33B,Q34B,Q35B]


ListeQLtext = ['Q7','Q9','Q10','Q12','Q14',
            'Q16','Q17','Q18','Q19','Q23',
            'Q26A','Q27A','Q28A','Q29A',
            'Q31A','Q33A','Q34A','Q35A',
            'Q36','Q37','Q39',
            'Q26B','Q27B','Q28B','Q29B',
            'Q31B','Q33B','Q34B','Q35B']

ListeQUtext = ['Q1','Q2','Q3','Q4','Q5','Q6','Q8',
           'Q11','Q13','Q15','Q20','Q21','Q22',
           'Q24','Q25','Q30','Q32','Q38']

ListeLabels = [LabelsdfQ7,LabelsdfQ9,LabelsdfQ10,LabelsdfQ12,LabelsdfQ14,
                LabelsdfQ16,LabelsdfQ17,LabelsdfQ18,LabelsdfQ19,LabelsdfQ23,
                LabelsdfQ26A,LabelsdfQ27A,LabelsdfQ28A,LabelsdfQ29A,
                LabelsdfQ31A,LabelsdfQ33A,LabelsdfQ34A,LabelsdfQ35A,
                LabelsdfQ36,LabelsdfQ37,LabelsdfQ39,
                LabelsdfQ26B,LabelsdfQ27B,LabelsdfQ28B,LabelsdfQ29B,
                LabelsdfQ31B,LabelsdfQ33B,LabelsdfQ34B,LabelsdfQ35B,]

dict_df = {Q7:dfQ7,Q9:dfQ9,Q10:dfQ10,Q12:dfQ12,Q14:dfQ14,
                Q16:dfQ16,Q17:dfQ17,Q18:dfQ18,Q19:dfQ19,Q23:dfQ23,
                Q26A:dfQ26A,Q27A:dfQ27A,Q28A:dfQ28A,Q29A:dfQ29A,
                Q31A:dfQ31A,Q33A:dfQ33A,Q34A:dfQ34A,Q35A:dfQ35A,
                Q36:dfQ36,Q37:dfQ37,Q39:dfQ39,
                Q26B:dfQ26B,Q27B:dfQ27B,Q28B:dfQ28B,Q29B:dfQ29B,
                Q31B:dfQ31B,Q33B:dfQ33B,Q34B:dfQ34B,Q35B:dfQ35B}

dict_text = {'Q7':Q7,'Q9':Q9,'Q10':Q10,'Q12':Q12,'Q14':Q14,
                'Q16':Q16,'Q17':Q17,'Q18':Q18,'Q19':Q19,'Q23':Q23,
                'Q26A':Q26A,'Q27A':Q27A,'Q28A':Q28A,'Q29A':Q29A,
                'Q31A':Q31A,'Q33A':Q33A,'Q34A':Q34A,'Q35A':Q35A,
                'Q36':Q36,'Q37':Q37,'Q39':Q39,
                'Q26B':Q26B,'Q27B':Q27B,'Q28B':Q28B,'Q29B':Q29B,
                'Q31B':Q31B,'Q33B':Q33B,'Q34B':Q34B,'Q35B':Q35B}

ListeQUtext = ['Q1','Q2','Q3','Q4','Q5','Q6','Q8',
           'Q11','Q13','Q15','Q20','Q21','Q22',
           'Q24','Q25','Q30','Q32','Q38']

dict_textU = {'Q1':Q1,'Q2':Q2,'Q3':Q3,'Q4':Q4,'Q5':Q5,'Q6':Q6,
                'Q8':Q8,'Q11':Q11,'Q13':Q13,'Q15':Q15,
                'Q20':Q20,'Q21':Q21,'Q22':Q22,'Q24':Q24,
                'Q25':Q25,'Q30':Q30,'Q32':Q32,'Q38':Q38}

dict_textU2 = {Q1:'Q1',Q2:'Q2',Q3:'Q3',Q4:'Q4',Q5:'Q5',Q6:'Q6',
                Q8:'Q8',Q11:'Q11',Q13:'Q13',Q15:'Q15',
                Q20:'Q20',Q21:'Q21',Q22:'Q22',Q24:'Q24',
                Q25:'Q25',Q30:'Q30',Q32:'Q32',Q38:'Q38'}

dict_df_text = {'Q7':dfQ7,'Q9':dfQ9,'Q10':dfQ10,'Q12':dfQ12,'Q14':dfQ14,
                'Q16':dfQ16,'Q17':dfQ17,'Q18':dfQ18,'Q19':dfQ19,'Q23':dfQ23,
                'Q26A':dfQ26A,'Q27A':dfQ27A,'Q28A':dfQ28A,'Q29A':dfQ29A,
                'Q31A':dfQ31A,'Q33A':dfQ33A,'Q34A':dfQ34A,'Q35A':dfQ35A,
                'Q36':dfQ36,'Q37':dfQ37,'Q39':dfQ39,
                'Q26B':dfQ26B,'Q27B':dfQ27B,'Q28B':dfQ28B,'Q29B':dfQ29B,
                'Q31B':dfQ31B,'Q33B':dfQ33B,'Q34B':dfQ34B,'Q35B':dfQ35B}

dict_labels = {Q7:LabelsdfQ7,Q9:LabelsdfQ9,Q10:LabelsdfQ10,Q12:LabelsdfQ12,Q14:LabelsdfQ14,
                Q16:LabelsdfQ16,Q17:LabelsdfQ17,Q18:LabelsdfQ18,Q19:LabelsdfQ19,Q23:LabelsdfQ23,
                Q26A:LabelsdfQ26A,Q27A:LabelsdfQ27A,Q28A:LabelsdfQ28A,Q29A:LabelsdfQ29A,
                Q31A:LabelsdfQ31A,Q33A:LabelsdfQ33A,Q34A:LabelsdfQ34A,Q35A:LabelsdfQ35A,
                Q36:LabelsdfQ36,Q37:LabelsdfQ37,Q39:LabelsdfQ39,
                Q26B:LabelsdfQ26B,Q27B:LabelsdfQ27B,Q28B:LabelsdfQ28B,Q29B:LabelsdfQ29B,
                Q31B:LabelsdfQ31B,Q33B:LabelsdfQ33B,Q34B:LabelsdfQ34B,Q35B:LabelsdfQ35B}



#*****************************************************************
# Création des labels propres à chaque question à choix multiples
#*****************************************************************

for i in range(len(ListeQL)):

    question = ListeQL[i]
    LabelQ = ListeLabels[i]

    for j in dict_df[question]:
            LabelQ.append(dict_df[question][j].value_counts().index[0])

    dict_df[question] = dict_df[question].set_axis(LabelQ, axis=1, inplace=False)

    for code in LabelQ:
            dict_df[question][code] = dict_df[question][code].replace(to_replace = [np.nan,code],value= [0,1]) 
            
#harmonisation des labels des questions A/B
LabelsdfQ27A[10]='None'
dict_df[Q27A] = dict_df[Q27A].set_axis(LabelsdfQ27A, axis=1, inplace=False)

LabelsdfQ28A[9]='None'
dict_df[Q28A] = dict_df[Q28A].set_axis(LabelsdfQ28A, axis=1, inplace=False)

LabelsdfQ33A[6]='None'
dict_df[Q33A] = dict_df[Q33A].set_axis(LabelsdfQ33A, axis=1, inplace=False)  

LabelsdfQ34A[10]='None'
dict_df[Q34A] = dict_df[Q34A].set_axis(LabelsdfQ34A, axis=1, inplace=False)          

LabelsdfQ35A[9]='None'
dict_df[Q35A] = dict_df[Q35A].set_axis(LabelsdfQ35A, axis=1, inplace=False)               

LabelsdfQ33B[5]='Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)'
dict_df[Q33B] = dict_df[Q33B].set_axis(LabelsdfQ33B, axis=1, inplace=False)        

#*****************************************************************
#*****************************************************************

#*****************************************************************
# Création des labels et subdataframes comme livrable
#*****************************************************************

labels_dfQ7=[]

for i in dfQ7:
        labels_dfQ7.append(dfQ7[i].value_counts().index[0])

dfQ7 = dfQ7.set_axis(labels_dfQ7, axis=1, inplace=False)

for code in labels_dfQ7:
        dfQ7[code] = dfQ7[code].replace(to_replace = [np.nan,code],value= [0,1])


labels_dfQ9=[]

for i in dfQ9:
        labels_dfQ9.append(dfQ9[i].value_counts().index[0])

dfQ9 = dfQ9.set_axis(labels_dfQ9, axis=1, inplace=False)

for code in labels_dfQ9:
        dfQ9[code] = dfQ9[code].replace(to_replace = [np.nan,code],value= [0,1])      
        
        
labels_dfQ14=[]

for i in dfQ14:
        labels_dfQ14.append(dfQ14[i].value_counts().index[0])


dfQ14 = dfQ14.set_axis(labels_dfQ14, axis=1, inplace=False)

for code in labels_dfQ14:
        dfQ14[code] = dfQ14[code].replace(to_replace = [np.nan,code],value= [0,1])          


labels_dfQ16=[]

for i in dfQ16:
        labels_dfQ16.append(dfQ16[i].value_counts().index[0])


dfQ16 = dfQ16.set_axis(labels_dfQ16, axis=1, inplace=False)

for code in labels_dfQ16:
        dfQ16[code] = dfQ16[code].replace(to_replace = [np.nan,code],value= [0,1]) 
        
        
dfQ7_pro = dfQ7[(df['Q5'] != 'Student')&(df['Q5'] != 'Other')&(df['Q5'] != 'Currently not employed')]
dfQ9_pro = dfQ9[(df['Q5'] != 'Student')&(df['Q5'] != 'Other')&(df['Q5'] != 'Currently not employed')]
dfQ14_pro = dfQ14[(df['Q5'] != 'Student')&(df['Q5'] != 'Other')&(df['Q5'] != 'Currently not employed')]
dfQ16_pro = dfQ16[(df['Q5'] != 'Student')&(df['Q5'] != 'Other')&(df['Q5'] != 'Currently not employed')]



#*********************************************************
# Questions à choix multiples : fusion des parties A et B 
#*********************************************************

dfQ26 = pd.DataFrame(dict_df[Q26A].sum().sort_index(),columns =['Professionals'])
dfQ26['Non-Professionals'] = dict_df[Q26B].sum().sort_index()
dfQ27 = pd.DataFrame(dict_df[Q27A].sum().sort_index(),columns =['Professionals'])
dfQ27['Non-Professionals'] = dict_df[Q27B].sum().sort_index()
dfQ28 = pd.DataFrame(dict_df[Q28A].sum().sort_index(),columns =['Professionals'])
dfQ28['Non-Professionals'] = dict_df[Q28B].sum().sort_index()
dfQ28 = pd.DataFrame(dict_df[Q28A].sum().sort_index(),columns =['Professionals'])
dfQ28['Non-Professionals'] = dict_df[Q28B].sum().sort_index()
dfQ29 = pd.DataFrame(dict_df[Q29A].sum().sort_index(),columns =['Professionals'])
dfQ29['Non-Professionals'] = dict_df[Q29B].sum().sort_index()

dfQ31 = pd.DataFrame(dict_df[Q31A].sum().sort_index(),columns =['Professionals'])
dfQ31['Non-Professionals'] = dict_df[Q31B].sum().sort_index()
dfQ33 = pd.DataFrame(dict_df[Q33A].sum().sort_index(),columns =['Professionals'])
dfQ33['Non-Professionals'] = dict_df[Q33B].sum().sort_index()
dfQ34 = pd.DataFrame(dict_df[Q34A].sum().sort_index(),columns =['Professionals'])
dfQ34['Non-Professionals'] = dict_df[Q34B].sum().sort_index()
dfQ35 = pd.DataFrame(dict_df[Q35A].sum().sort_index(),columns =['Professionals'])
dfQ35['Non-Professionals'] = dict_df[Q35B].sum().sort_index()

dict_df_graphAB = {Q26A:dfQ26,Q27A:dfQ27,Q28A:dfQ28,Q29A:dfQ29,
                Q31A:dfQ31,Q33A:dfQ33,Q34A:dfQ34,Q35A:dfQ35}

#*********************************************************



#*********************************
# Mise en page de l' INTRODUCTION
#*********************************

if option == 'Introduction':

    with titre:
        
        col1, col2, col3 = st.columns(3)

        #with col1:

        with col2:
        
            st.image("DataScientest.jpg")
            st.markdown(" > > > # PyGraal  ")
            st.markdown(" > > > ## Data Jobs")
        
        #with col3:
        
        st.title("")    
        st.title("") 
    

    with intro:
    
        st.header("Objectif")
        st.markdown("L’objectif de ce projet est de comprendre à l’aide des données les différents profils techniques qui se sont créés dans l’industrie de la Data. Plus exactement, il faudra mener une analyse poussée des tâches effectuées ainsi que des outils utilisés par chaque poste afin d’établir des ensembles de compétences et outils correspondant à chaque poste du monde de la Data.") 
        st.markdown("Ensuite, il sera possible de construire un système de recommandation de poste permettant aux autres apprenants de viser le poste correspondant le plus à leurs appétences.")
        st.markdown("C’est un sujet d’autant plus intéressant que nous le retrouvons régulièrement dans la presse spécialisée.")
        st.text("")
        st.text("")
        
        st.header("Méthodologie")
        st.markdown("Voici la démarche que nous avons suivie:")
        st.markdown("1.  Présentation du jeu de données\n"
                    "> - Analyse exploratoire\n"
                    "> - Data Visualization\n"
                    "\n2.  Prétraitement des données\n"
                    "> - Nettoyage en plusieurs étapes\n"
                    "\n3.  Stratégie et modélisation\n"
                    "> - Comparaison de plusieurs modèles\n"
                    "> - Tests de Réduction de dimensions\n"
                    "> - Tests de Ré-échantillonnage\n"
                    "> - Optimisation des hyperparamètres\n"
                    "\n4.  Démo\n"
                    "\n5.  Conclusion et perspectives")
        
        
        st.caption("*Les différentes parties son accessibles grâce au sommaire sur la gauche de l'application.")

        st.image("01_acculturation_data_pipeline.jpg", width =1100) 

#*********************************


#*********************************************************
# Mise en page du chapitre PRESENTATION DU JEU DE DONNEES
#*********************************************************

if option == 'Présentation du jeu de données':
    
    with dataset:

        st.header("Jeu de données")
        st.markdown("Cette analyse repose sur les réponses d’un sondage auprès de 20 037 personnes provenant de 171 pays différents. ")
        st.markdown("https://www.kaggle.com/c/kaggle-survey-2020/overview")
        st.markdown("Les participants ont dû répondre à une série d'une quarantaine de questions. Quelques-unes sont d'ordre personnel (âge, genre...) ou bien en rapport avec leur poste et entreprise actuels. Cependant, l'essentiel concerne avant tout les outils utilisés dans le domaine de la Data Science (langage de programmation, bibliothèque de data visualization...) ainsi que le rapport à l'utilisation de Machine Learning. ")
        st.markdown("Ce sondage comprend à la fois des questions à choix unique et à choix multiples. A noter également que certaines questions étaient accessibles en fonction de réponses précédentes. ")    
        st.markdown("A partir de ce sondage, nous souhaitons établir des ensembles de compétences et outils correspondant aux divers postes du monde de la Data, afin de construire dans un second temps un système de recommandation de poste permettant au demandeur de viser exactement le poste correspondant le plus à ses appétences.")
        st.text("")
        
       

        ListeQ = ['---',Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,
                 Q10,Q11,Q12,Q13,Q14,Q15,Q16,Q17,Q18,Q19,
                 Q20,Q21,Q22,Q23,Q24,Q25,Q26A,Q27A,Q28A,Q29A,
                 Q30,Q31A,Q32,Q33A,Q34A,Q35A,Q36,Q37,Q38,Q39]
        
        Liste_notes = [Q18,Q19,Q27A,Q28A,Q30,Q32,Q34A]
        
        Liste_notes2 = [Q26A,Q27A,Q28A,Q29A,Q31A,Q33A,Q34A,Q35A]
                
        ListeSelectSlider = [Q24]
        ListeRevenus = ['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999',
                '10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999',      
                '50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999',      
                '100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999',    
                '300,000-500,000','> $500,000']
        
        ListeSelect = [Q3]
        ListeRadio = [Q1,Q2,Q4,Q5,Q6,Q8,Q11,Q13,Q15,Q20,Q21,
                      Q22,Q25,Q30,Q32,Q38]
        ListeMulti = [Q7,Q9,Q10,Q12,Q14,Q16,Q17,Q18,Q19,Q23,Q26A,
                      Q27A,Q28A,Q29A,Q31A,Q33A,Q34A,Q35A,Q36,Q37,Q39]
        
        
        #*********************************************************
        # Aperçu des questions et des choix de réponses possibles 
        #*********************************************************
        
        st.subheader("Présentation du questionnaire")
    
        question = st.selectbox('QUESTION',ListeQ)       

        if question == ListeQ[0]:
            st.text("")    
        
        elif question in ListeSelectSlider:
            responses = st.select_slider("Revenus", options=ListeRevenus,value='30,000-39,999')
        
        elif question in ListeSelect:
            responses = st.selectbox("PAYS",Countries)
        
        elif question in ListeRadio:
            if question in Liste_notes:
                st.caption(dict_notes[question])
            if question in Liste_notes2:
                st.caption(dict_notes2[question])
            responses = st.radio("",dict_quest[question])

        elif question in ListeMulti:
            if question in Liste_notes:
                st.caption(dict_notes[question])
            if question in Liste_notes2:
                st.caption(dict_notes2[question])
            responses = st.multiselect("",dict_quest[question])
       
        else:
            st.image("WorkInProgress.jpg")    
            
        #*********************************************************

        #******************************    
        # Aperçu des réponses obtenues
        #******************************    

        st.text("")
        
        if question == ListeQ[0]:
            st.text("")    
    
        elif question in ListeQU:
            BarChart = pd.DataFrame(df[dict_textU2[question]].value_counts())
            st.markdown("**Aperçu des réponses**")
            st.bar_chart(BarChart)
 
        elif question in ListeQM:
            BarChart = pd.DataFrame(dict_df[question].sum().sort_values())
            st.markdown("**Aperçu des réponses**")
            st.bar_chart(BarChart)
         
        elif question in ListeQMab:
            BarChart = pd.DataFrame(dict_df_graphAB[question][['Professionals','Non-Professionals']])
            st.markdown("**Aperçu des réponses**")
            st.bar_chart(BarChart)
        
        else:
            st.image("WorkInProgress.jpg")        
                 
        #******************************    

        #*********************************************************
        # Quelques exemples de datavisualization
        #*********************************************************
      
        st.header("Data Visualization")
        st.header("")
        st.markdown("Afin d'explorer ce jeu de données, nous nous sommes posés des questions et avons représenté ces résultats.")
        st.markdown("")

        #****************************************************************
        st.subheader("**Le domaine de la Data serait réservé aux hommes et ne regrouperait que quatre ou cinq métiers ?**")


        df_sx = df_pro[(df_pro['Q2'] == 'Man')|(df_pro['Q2'] == 'Woman')]

        #A voir pour réussir à centrer le titre (utilisations des 3 colonnes?)

        colViz1, colViz2 = st.columns([3,1])
        #TypeGraph='"Par genre"'

        with colViz2:
            TypeGraph = st.radio("Répartition",["Par genre","Par métier"])

        with colViz1:
            if TypeGraph == 'Par genre':
                var1 = 'Q2'
                var2 = 'Q5'
                TitreViz = '> > > > > > > > > > **Répartition des métiers de la Data par genre**'
            else:
                var1 = 'Q5'
                var2 = 'Q2'
                TitreViz = '> > > > > > > > > > > **Répartition des genres par métier de la Data**'
            
            fig = px.sunburst(
                data_frame=df_sx,
                path = [var1,var2],
                color = var2,
                color_discrete_sequence = px.colors.qualitative.Pastel,
                maxdepth = -1,
                hover_name=var1,
                hover_data={var2 : False},
                )
            fig.update_traces(textinfo= 'label+percent parent',
                              texttemplate='<i>%{label}</i> <br> %{percentParent}',
                              )
            fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
            
            st.plotly_chart(fig)
            st.markdown(TitreViz)  
        st.text("")
        #Conclusion de la visualization
        st.markdown("Tout d'abord, nous constatons un déséquilibre flagrant des genres qui semble confirmer que les métiers de la Data sont fortement masculinisés.\n"
                    "Par ailleurs, en se focalisant sur certains métiers comme Product Manager ou Machine Learning Engineers, ce constat est encore plus marqué.\n"
                    "En ce qui concerne les métiers, il y en a une dizaine, mais dont trois représentent plus de la moitié du panel, à savoir *Data Scientist*, *Software Engineer* et *Data Analyst*."
                    )
        
        #****************************************************************
        
        st.header("")
        st.subheader("**Quels sont les 'outils' les plus utilisés dans les métiers de la Data?**")
        st.markdown("Afin d'illustrer de la manière la plus synthétique notre hypothèse, nous n'avons retenu que quatre questions portant respectivement sur l'utilisation quotidienne de langage(s) de programmation, d'environnement(s), d'outils de Data visualization et de machine learning. Ces réponses sont-elles en adéquation avec le contenu enseigné par DataScientest ? :smile:")
        st.markdown("")
        st.caption("Cocher la question afin de connaitre les réponses du panel.")
               
        Viz1 = st.checkbox(Q7,False)
        if Viz1 == 1:
            qViz1=Q7
            
            #Graph Langage programmation Q7
            dfQ7_graph = pd.DataFrame({'Programming languages' : dfQ7_pro.sum().sort_values().index , 'Count' : dfQ7_pro.sum().sort_values()}).reset_index()
            dfQ7_graph.drop('index', axis=1, inplace=True)
            fig = px.bar(dfQ7_graph, x="Programming languages", y="Count", orientation='v', color_discrete_sequence = ['#4169E1'])
            
            st.write(fig)
        
        else:
            st.text("")
        
        Viz2 = st.checkbox(Q9,False)
        if Viz2 == 1:
            qViz2=Q9
            
            #Graph IDE Q9
            dfQ9_graph = pd.DataFrame({'IDE' : dfQ9_pro.sum().sort_values().index , 'Count' : dfQ9_pro.sum().sort_values()}).reset_index()
            dfQ9_graph.drop('index', axis=1, inplace=True)
            dfQ9_graph['IDE'].replace(['Jupyter (JupyterLab, Jupyter Notebooks, etc) ', 'Visual Studio Code (VSCode)'],['Jupyter', 'VSCode'], inplace=True)
            fig = px.bar(dfQ9_graph, x="IDE", y="Count", orientation='v', color_discrete_sequence = ['#2E8B57'])
            
            st.write(fig)

        else:
            st.text("")
            
        Viz3 = st.checkbox(Q14,False)
        if Viz3 == 1:
            qViz3=Q14
            
            #Graph Data viz
            dfQ14_graph = pd.DataFrame({'Data Visualization libraries' : dfQ14_pro.sum().sort_values().index , 'Count' : dfQ14_pro.sum().sort_values()}).reset_index()
            dfQ14_graph.drop('index', axis=1, inplace=True)
            fig = px.bar(dfQ14_graph, x="Data Visualization libraries", y="Count", orientation='v', color_discrete_sequence = ['#FFB233'])
            
            st.write(fig)
       
        else:
            st.text("")
            
        Viz4 = st.checkbox(Q16,False)
        if Viz4 == 1:
            qViz4=Q16
            
            #Graph ML frameworks
            dfQ16_graph = pd.DataFrame({'ML frameworks' : dfQ16_pro.sum().sort_values().index , 'Count' : dfQ16_pro.sum().sort_values()}).reset_index()
            dfQ16_graph.drop('index', axis=1, inplace=True)
            fig = px.bar(dfQ16_graph, x="ML frameworks", y="Count", orientation='v', color_discrete_sequence = ['#BB33FF'])
            
            st.write(fig)
    
        else:
            st.text("")

        st.markdown("Voici ce que nous pouvons retenir de ces graphes:\n"
                    "- **Python** est de loin le langage de programmation le plus populaire dans les métiers de la data. Il semble donc être bien le **Graal** pour travailler dans ce domaine.\n"
                    "- Jupyter Notebook est l'IDE privilégié, bien que d'autres IDE soient aussi utilisés.\n"
                    "- Quand on parle de data visualization, Matplotlib et Seaborn se détachent clairement des autres outils à disposition.\n"
                    "- Enfin le Machine Learning se fait majoritairement via Scikit-learn, même si TensorFlow et Keras ont aussi un certain succès."
                    )
        st.markdown(":fast_forward: Le contenu de formation de DataScientest semble refléter de façon assez correcte, les outils actuellement utilisés par les professionnels de la Data.")


#*********************************************************   
        
        st.header("")
        st.subheader("**Big Data = Big Company ?**")
        st.markdown("Le Machine Learning n'est-il utilisé que dans les grandes entreprises ?")
        st.markdown("")

        df_taille = df_pro[['Q20', 'Q22']]
        df_taille['COUNTER'] = 1
        rep1 = df_taille.groupby(['Q20','Q22'])['COUNTER'].sum()
        rep2 = rep1.groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()),2)).reset_index()
        Q20_mod = rep2['Q20'].replace(['0-49 employees', '10,000 or more employees',
               '1000-9,999 employees', '250-999 employees', '50-249 employees'], ['0 à 49', '> 10 000', '1000 à 9,999', '250 à 999', '50 à 249'])
        Qorder = rep2['Q20'].replace(['0-49 employees', '10,000 or more employees',
               '1000-9,999 employees', '250-999 employees', '50-249 employees'], [1, 5, 4, 3, 2])
        
        rep2['Q20_mod']=Q20_mod
        rep2['Qorder']=Qorder
        rep2 = rep2.sort_values(by='Qorder')
        
        
        Q22_mod = rep2['Q22'].replace(['I do not know', 'No (we do not use ML methods)',
               'We are exploring ML methods (and may one day put a model into production)',
               'We have well established ML methods (i.e., models in production for more than 2 years)',
               'We recently started using ML methods (i.e., models in production for less than 2 years)',
               'We use ML methods for generating insights (but do not put working models into production)'],[0,1,3,5,4,2])
        rep2['ML use Scale']=Q22_mod
        
        fig = px.line(rep2, x="Q20_mod", y="COUNTER", color='ML use Scale', markers = True, symbol = 'ML use Scale',
                      labels={"Q20_mod": "Company size",  "COUNTER": "Ratio (by company size)"},
#                      legend={'traceorder':'normal'},
                      )
        fig.update_traces(textposition="bottom right", hovertemplate=None)
        fig.update_layout(margin=dict(t=0, b=0, r=0, l=0), hovermode="x")
        #fig.update_xaxes(categoryarray = ["Qorder"])
        
#        st.markdown("# Répartition des usages du Machine Learning par taille d'entreprise")
#        st.markdown("## Répartition des usages du Machine Learning par taille d'entreprise")
#        st.markdown("### Répartition des usages du Machine Learning par taille d'entreprise")
        st.markdown("> > > > > **Répartition des usages du Machine Learning par taille d'entreprise**")
        
        st.write(fig)
        st.markdown('')
         
        st.markdown("> > > >*Tableau de correspondance entre les réponses obtenues et l'échelle affichée*")
        fig = go.Figure(data = [go.Table(
            columnwidth = [1, 3],
            header = dict(values = ['ML use Scale', 'Category']),
            #height = 40,
            cells = dict(values = [[0,1,2,3,4,5],['I do not know',
                                                  'No (we do not use ML methods)',
                                                  'We use ML methods for generating insights (but do not put working models into production)',
                                                  'We are exploring ML methods (and may one day put a model into production)',
                                                  'We recently started using ML methods (i.e., models in production for less than 2 years)',
                                                  'We have well established ML methods (i.e., models in production for more than 2 years)']],
            align = 'center',
            height = 30))])    
        fig.update_layout(margin = dict(l = 5, r = 5, b = 10, t = 10),
                         width=650, height=250)
        
        st.write(fig)        
        
        st.markdown("On voit clairement que plus la taille de l'entreprise est grande, plus les modèles de Machine Learning sont non seulement en production depuis plus de 2 ans, mais aussi ancrés dans la culture d'entreprise.\n"
                    "On constate toutefois que le Machine Learning est en cours de développement quelle que soit la taille de l'entreprise.")
        
#*********************************************************   



    
#****************************************************
# Mise en page du chapitre PRETRAITEMENT DES DONNEES
#****************************************************

### Mise en page de la page Preprocessing

if option == 'Prétraitement des données':

    with preprocessing:

        st.header("Etape 0 / Etape 1")        

        st.subheader("Analyse et nettoyage des doublons et valeurs manquantes")
        st.markdown("Nous constatons que les différents postes en Data Science se trouve dans la colonne Q5 qui correspond donc à notre variable cible pour la suite de l'étude.") 
        st.markdown("14 doublons sont identifiés et supprimés.")
        st.markdown("745 valeurs manquantes sont identifiées et supprimées pour notre variable cible.")
        st.markdown("Il n'y a que des valeurs catégorielles, un encodage sera nécessaire pour la création d'un modèle de Machine Learning.")
        st.markdown("La première ligne du DataFrame n'est pas une entrée mais correspond à l'intitulé des différentes questions. Cette entrée n'est donc pas à exploiter mais elle reste intéressante pour la compréhension du jeu de données. Nous choisissons de supprimer cette première ligne, mais la gardons à disposition pour s'y référer au cours de notre étude.")
        st.markdown("La première colonne représente le temps de réponse au questionnaire. Nous la supprimons car elle n'est pas pertinente pour notre étude.")

        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure (figsize = (15,7))
        sns.heatmap(df.isna(), cbar = False);
        st.pyplot()
        
        st.markdown("Cette visualisation montre les valeurs manquantes en beige, il est clair que celles-ci sont très nombreuses, même majoritaires.")
        st.markdown("Dans cet état, nous ne pouvons appliquer une méthode de nettoyage des valeurs manquantes sans approfondir notre analyse.")
        st.markdown("Nous décidons donc de séparer les questions en 2 sous ensembles en fonction du type de question: à choix unique et à choix multiples.")        
        
        st.subheader("Traitement des questions à choix unique")
        st.markdown("Nous avons 8 questions de ce type, leur traitement est relativement simple.\n"
                    "Après analyse de ces données, nous remarquons que le taux de nans est faible, nous remplaçons donc ces valeurs manquantes par les modes des variables correspondantes.")
        
        st.subheader("Traitement des questions à choix multiples")
        st.markdown("Les questions à choix multiples sont plus difficiles à interpréter de par la manière dont elles ont été intégrées dans la base de données.")
        st.markdown("Pour chaque question, un nombre de colonnes équivalent au nombre de réponses proposées a été créé.\n"
                    "Pour chaque réponse, chaque colonne est remplie de la valeur de la réponse si le répondant a sélectionné ce choix et de NaNs sinon. Il ne faut pas supprimer les valeurs manquantes sur ces colonnes car elles ont une utilité de compréhension.")
        st.markdown("Afin de pouvoir les utiliser et les analyser ultérieurement, il est cependant nécessaire de les encoder de manière binaire : 1 en cas de réponse positive, 0 si la valeur est manquante.")
        
        st.markdown("Ci-dessous un aperçu du travail réalisé:")
        
        Encoded = st.select_slider("Question à choix multiples", ListeQLtext)
        Texte = dict_text[Encoded]
        
        st.markdown("Données brutes")
        st.table(dict_df_text[Encoded].head())
        st.markdown("Données encodées")
        st.table(dict_df[Texte].head())


        st.header("Etape 1 / Etape 2")        

        st.subheader("Analyse de la variable cible")
        st.markdown("Comme vu précédemment, la valeur cible est 'Q5' (poste actuellement occupé) et après un premier traitement, elle contient 10 valeurs uniques.")
        st.markdown("Le domaine de la Data est en constante évolution, ses métiers également. De ce fait, nous allons restreindre notre analyse au 5 principaux métiers, qui représentent à eux seuls près de 80% du panel professionnel interrogé.")
        st.markdown("Pour appuyer cette réflexion, nous nous sommes inspirés d'articles comme celui-ci précisant qu'au sein même du métier de Data Scientist il y avait des différenciations:")
        st.markdown("https://www.journaldunet.com/solutions/dsi/1506951-de-l-avenir-du-metier-de-data-scientist-e/")
        
        st.subheader("Traitement des features")
        st.markdown("Pour rappel, notre objectif est de créer un modèle capable de proposer un poste en fonction de compétences et d'outils associés. Par conséquent, en analysant les questions posées, nous pouvons supprimer une partie des colonnes.")
        st.markdown("Voici la liste des colonnes concernées et notre raisonnement:")

        fig = go.Figure(data = go.Table(columnwidth = [2, 4, 6],
            header = dict(values = list(table_quest[['Q', 'Theme_de_la_question', 'Raisonnement']].columns),
                         height = 40),
            cells = dict(values = [table_quest.Q, table_quest.Theme_de_la_question, table_quest.Raisonnement],
                        align = 'left',
                        height = 50)))        
        fig.update_layout(margin = dict(l = 5, r = 5, b = 0, t = 10),
                         width=1000, height=400)
        st.write(fig)
        
        st.markdown("**Encodage des colonnes restantes**")
        st.markdown("L'ensemble des questions à choix multiples a déjà été traité précédemment. Il nous reste à encoder Q6 et Q15:\n"
                    "- Q6 - For how many years have you been writing code and/or programming?\n"
                    "- Q15 - For how many years have you used machine learning methods?")

        st.header('')
        
        st.subheader("Résumé du nettoyage de la base de données")
    
        Etapes = ['initial','professionnels','nettoyé']
        Steps = st.select_slider("JEU DE DONNEES", Etapes,'initial')
    
        col4, col5, col6 = st.columns(3)
        col11,col12,col13,col14,col15 = st.columns(5)
    
        if Steps== 'initial':
        
            with col4:
                st.markdown("**Etape 0**")
                st.markdown("Jeu de données initial :")
                st.markdown(str(lignes) + " lignes, " + str(colonnes) + " colonnes")
                st.markdown("*Plus de 13 métiers*")
                
            with col11:    
                comptage = df['Q5'].value_counts()
                values = comptage.tolist()
                names = comptage.index.tolist()
                print(names)

                css_color =['#F5FFFA', '#B0E0E6', '#FFF0F5', '#E6E6FA', '#F08080', '#FFFACD', '#B0C4DE',
                            '#8FBC8F', '#9370DB', '#FFB6C1', '#E0FFFF', '#90EE90', '#FF7F50']

                fig = px.pie(df, values = values, names = names,
                             color = names,
                              color_discrete_sequence = css_color,
                             )
                fig.update_traces(textinfo= 'percent+label',
                                  showlegend = False,
                                  hovertemplate = "<b>%{label}:</b> <br> <i>%{value}</i> </br> %{percent:.2%f}",
                                  texttemplate='<b>%{label}:</b> <br> %{percent:.2%f}',
                                  hole = 0.2,
                                  )
                fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
                st.write(fig)
        
        elif Steps == 'professionnels':
        
            with col5:
                st.markdown("**Etape 1**")
                st.markdown("Jeu de données 'professionnels' :")
                st.markdown(str(df_pro.shape[0]) + " lignes, " + str(df_pro.shape[1]) + " colonnes")
                st.markdown("*10 métiers*")

            with col12:
                comptage = df_pro['Q5'].value_counts()
                values = comptage.tolist()
                names = comptage.index.tolist()
                print(names)

                css_color =['#B0E0E6', '#FFF0F5', '#FFFACD', '#B0C4DE',
                            '#8FBC8F', '#9370DB', '#FFB6C1', '#E0FFFF', '#90EE90', '#FF7F50']

                fig = px.pie(df, values = values, names = names,
                             color = names,
                              color_discrete_sequence = css_color,
                             )
                fig.update_traces(textinfo= 'percent+label',
                                  showlegend = False,
                                  hovertemplate = "<b>%{label}:</b> <br> <i>%{value}</i> </br> %{percent:.2%f}",
                                  texttemplate='<b>%{label}:</b> <br> %{percent:.2%f}',
                                  hole = 0.2,
                                  )
                fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
                st.write(fig)
    
        else:
        
            with col6:
                st.markdown("**Etape 2**")
                st.markdown("Jeu de données nettoyé :")
                st.markdown(str(df_clean.shape[0]) + " lignes, " + str(df_clean.shape[1]) + " colonnes")
                st.markdown("*5 métiers*")
                
            with col13:
                comptage = df_clean['Q5'].value_counts()
                values = comptage.tolist()
                names = comptage.index.tolist()
                print(names)

                css_color =['#B0E0E6', '#FFF0F5', '#FFFACD', '#B0C4DE', '#8FBC8F']

                fig = px.pie(df, values = values, names = names,
                             color = names,
                              color_discrete_sequence = css_color,
                             )
                fig.update_traces(textinfo= 'percent+label',
                                  showlegend = False,
                                  hovertemplate = "<b>%{label}:</b> <br> <i>%{value}</i> </br> %{percent:.2%f}",
                                  texttemplate='<b>%{label}:</b> <br> %{percent:.2%f}',
                                  hole = 0.2,
                                  )
                fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
                st.write(fig)
 

        st.caption("Etape 0 ==> Etape 1 ~ suppressions de lignes")
        st.caption("Etape 1 ==> Etape 2 ~ suppression de colonnes")
        
#************************************************************************************

     

#****************************************************
# Mise en page du chapitre STRATEGIE ET MODELISATION
#****************************************************    

if option == 'Stratégie et Modélisation':
        
    with MLalgorithm:
    
        st.header('Stratégie & Modélisation')
        st.markdown("A partir du dataset nettoyé et réduit aux colonnes pertinentes, nous avons:")
        st.markdown('1.   défini les features et la variable cible\n'
                    '2.   réparti le jeu de données en train set et test set\n'
                    '3.   standardisé ces données')

        st.markdown('Nous avons ensuite entraîné et mesuré les performances de 5 modèles, à savoir :')
        st.markdown('*   Decision Tree :arrow_right: *dtc*\n'
                    '*   Logistic Regression :arrow_right: *lr*\n'
                    '*   K Nearest Neighbors :arrow_right: *knn*\n'
                    '*   Support Vector Machine :arrow_right: *svm*\n'
                    '*   Random Forest :arrow_right: *rf*')

        st.markdown('')

        st.subheader('Comparaison des scores des 5 modèles')

        Data = {'Modèle' : ['dtc', 'dtc', 'lr', 'lr', 'knn', 'knn', 'svm', 'svm', 'rf', 'rf'],
                'Score' : ['Train', 'Test','Train', 'Test','Train', 'Test','Train', 'Test','Train', 'Test'],
                'Valeurs' : [0.9750, 0.4036, 0.5825, 0.5430, 0.6313, 0.3820, 0.7425, 0.5433, 0.9751, 0.5337]
                }

        Scores = pd.DataFrame(Data)

        fig = px.line(Scores, x="Modèle", y="Valeurs", color="Score",
                      color_discrete_sequence = ["red", "blue"],
                      range_y=([0.3,1]),
                      )
        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_layout(hovermode="x")
        st.plotly_chart(fig)

        #Conclusion comparaison 5 modèles
        #*********************************
        st.markdown('Les résultats des différents modèles entraînés présentent des disparités flagrantes.\n'
                    '\n"\U0001F50E" Les **scores de test** restent cantonnés entre **38 et 55%**, une performance relativement faible.\n'
                    '\n"\U0001F50E" Les scores d’entraînement mettent en évidence du **sur apprentissage** sur 4 des modèles (en particulier Arbre de Décision et Forêts Aléatoires, environ 97%).\n'
                    '\n"\U0001F50E" On distingue **3 groupes de modèles** : DTC et KNN à très faibles scores tests et sur apprentissage important, SVM et RF à sur apprentissage important, et LR.')

        st.subheader('')
        
        st.markdown(':fast_forward:  Nous avons donc retenu les trois modèles *Logistic Regression*, *Support Vector Machine*, et *Random Forest*.')
        st.subheader('')
        st.markdown("Ensuite, nous avons cherché à corriger le phénomène d'overfitting et à améliorer les performances au travers de trois étapes :\n"
                    "\na)   Réduction de dimensions\n"
                    "\nb)   Ré-échantillonnage\n"
                    "\nc)   Optimisation des hyperparamètres")
        st.subheader('')
        st.markdown("*Avant de présenter les résultats de ces tests, il est intéressant de préciser que l'importance/coefficient des variables sur chaque modèle est globalement faible.*")
        st.markdown("*Pour exemple, les deux variables 'Q6' et 'Q15' présentent les meilleurs coefficients avec des valeurs respectives de 0.047 et 0.041.*")
        st.header('')

        st.subheader('Tests de Réduction de dimensions') 
        st.markdown('Pour ces tests, nous avons utilisé les sélecteurs suivants :\n'
                    "*   SelectKBest avec score_func='mutual_info_classif' :arrow_right: *skb_mic*\n"
                    "*   SelectKBest avec score_func='f_classif' :arrow_right: *skb_fc*\n"
                    '*   SelectFromModel :arrow_right: *sfm*\n'
                    '*   Principal Components Analysis :arrow_right: *pca*\n'
                    '*   RFECV :arrow_right: *rfecv*')        


        reduc_lr, reduc_svm, reduc_rf = st.columns(3)

        with reduc_lr:
            st.image("Réduc_dim_lr.png")
        with reduc_svm:
            st.image("Réduc_dim_svm.png")
        with reduc_rf:
            st.image("Réduc_dim_rf.png")

        st.markdown('SelectFromModel offre le meilleur compromis entre le nombre de variables retenues et les scores obtenus, tout en réduisant le sur apprentissage.'
                    ' Une recherche du k optimum pour le SelectKBest aurait peut-être pu obtenir des résultats comparables ou meilleurs que le SelectFromModel.\n'
                    '\nPour le modèle Random Forest, nous n’avons pas réussi à trouver de compromis sans détériorer le score test qui a souffert de nos efforts de réduction du sur apprentissage.'
                    ' Des méthodes plus adaptées aux classifieurs de type arbre existent sûrement, nous n’avons pas su les identifier.')
        st.header('')

        st.subheader('Tests de Ré-échantillonnage')
        st.markdown('Pour ces tests, nous avons utilisé les ré-échantillonneurs suivants :\n'
                    "*   RandomUnderSampler :arrow_right: *rus*\n"
                    "*   ClusterCentroids :arrow_right: *cc*\n"
                    '*   Class_Weight défini manuellement :arrow_right: *cwm*\n'
                    '*   Class_Weight_Balanced :arrow_right: *cwb*\n'
                    '*   RandomOverSampler :arrow_right: *ros*\n'
                    '*   SMOTE :arrow_right: *smo*')             
        resamp_lr, resamp_svm, resamp_rf = st.columns(3)

        with resamp_lr:
            st.image("Resamp_lr.png")
        with resamp_svm:
            st.image("Resamp_svm.png")
        with resamp_rf:
            st.image("Resamp_rf.png")

        st.markdown('Depuis le début de nos tests, le modèle de régression logistique présente un faible écart entre les scores train et test.\n'
                    '\nPour SVM, on remarque l’amélioration des scores avec l’application d’un ré-échantillonneur par pondération manuelle.\n'
                    '\nEn ce qui concerne Random Forest, nous ne sommes pas parvenus à corriger le sur apprentissage. De la même manière que pour le redimensionnement, il existe peut-être des outils adaptés à ce type de classifieur.')

        st.markdown('')

        st.subheader('Optimisation des hyperparamètres')

        st.markdown('Après diverses tentatives pour causes principales de temps de calculs conséquents ou encore de méconnaissances de paramètres, voici les optimisations retenues et appliquées :')
        
        fig = go.Figure(data=[go.Table(
                 cells=dict(values=[['<b>Modèle</b>','<b>Réduction de dimensions</b>', '<b>Ré-échantillonnage</b>','<b>Optimisation des hyperparamètres</b>'],
                                    ['<i><b>Logistic Regression</i></b>','RFECV avec cv = 5','class_weight manuel','gridSearchCV avec cv = 5'],
                                    ['<i><b>Support Vector Machine</i></b>','SelectFromModel','Non appliqué','kernel = ‘linear’'],
                                    ['<i><b>Random Forest</i></b>','RFECV avec cv calculé sur KFold','Non appliqué','GridSearchCV avec cv calculé sur KFold']],
                            #height = 50,
                            ))])

        fig.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))
        fig.layout['template']['data']['table'][0]['header']['fill']['color']='rgba(0,0,0,0)'
        fig.update_layout(margin = dict(l = 5, r = 5, b = 0, t = 10),
                         width=900, height=200)
        st.write(fig)
        

        st.subheader('Comparaison des scores obtenus après optimisation')

        Data2 = {'Modèle' : ['lr', 'lr', 'lr', 'lr', 'svm', 'svm', 'svm', 'svm', 'rf', 'rf', 'rf', 'rf'],
                 'Score' : ['Train', 'Test', 'Train_opt', 'Test_opt', 'Train', 'Test', 'Train_opt', 'Test_opt', 'Train', 'Test', 'Train_opt', 'Test_opt'],
                 'Valeurs' : [0.5825, 0.5430, 0.5425, 0.5254, 0.7425, 0.5433, 0.5637, 0.5522, 0.9751, 0.5337, 0.8137, 0.5391]
             }

        Scores2 = pd.DataFrame(Data2)

        fig = px.line(Scores2, x="Modèle", y="Valeurs", color="Score",
                      color_discrete_sequence = ["red", "blue","red", "blue"],
                      line_dash = ['line', 'line', 'dot', 'dot','line', 'line', 'dot', 'dot','line', 'line', 'dot', 'dot'],
                      range_y=([0.3,1]),
                      )
        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig)

        st.markdown('')

        st.markdown("En conclusion de ces itérations, nous retenons que:\n"
                    "* Pour le modèle Logistic Regression :arrow_right: Nous avons pu réduire l’overfitting au détriment d’une légère détérioration de performance.\n"
                    "* Pour le modèle Support Vector Machine :arrow_right: L’overfitting a été corrigé et nous avons obtenu une petite amélioration de la performance.\n"
                    "* Pour le modèle Random Forest :arrow_right: Bien que considérablement réduit, l’overfitting reste important. La performance est équivalente.")
        st.markdown("Il est possible que nous n’ayons pas su trouver les sélecteurs et transformeurs les plus adaptés au modèle Random Forest pour obtenir de meilleurs résultats.")

#****************************************************
            

#*******************************
# Mise en page du chapitre DEMO
#*******************************

#import pandas as pd
#import streamlit as st
#import plotly.graph_objects as go
#import plotly.express as px

#st.set_page_config(layout="wide")

#df_clean = pd.read_csv('df_clean.csv')
#table_quest = pd.read_csv('table_quest.csv', index_col=0)
    
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if option == 'Démo':
  
    #Définition des questions à partir des features
    #***********************************************
    f6 = 'Q6'
    f15 = 'Q15'
    f7a = 'Q7:Python'
    f7b =  'Q7:R'
    f7c =  'Q7:SQL'
    f7d =  'Q7:C'
    f7e =  'Q7:C++'
    f7f =  'Q7:Java'
    f7g = 'Q7:Javascript'
    f7h = 'Q7:Julia'
    f7i = 'Q7:Swift'
    f7j = 'Q7:Bash'
    f7k = 'Q7:MATLAB'
    f7l = 'Q7:None'
    f7m = 'Q7:Other'
    f7n = 'Q9:Jupyter (JupyterLab Jupyter Notebooks etc) '
    f9a = 'Q9: RStudio '
    f9b = 'Q9:Visual Studio'
    f9c = 'Q9:Visual Studio Code (VSCode)'
    f9d = 'Q9: PyCharm '
    f9e = 'Q9:  Spyder  '
    f9f = 'Q9:  Notepad++  '
    f9g = 'Q9:  Sublime Text  '
    f9h = 'Q9:  Vim / Emacs  '
    f9i = 'Q9: MATLAB '
    f9j = 'Q9:None'
    f9k = 'Q9:Other'
    f10a = 'Q10: Kaggle Notebooks'
    f10b = 'Q10:Colab Notebooks'
    f10c = 'Q10:Azure Notebooks'
    f10d = 'Q10: Paperspace / Gradient '
    f10e = 'Q10: Binder / JupyterHub '
    f10f = 'Q10: Code Ocean '
    f10g = 'Q10: IBM Watson Studio '
    f10h = 'Q10: Amazon Sagemaker Studio '
    f10i = 'Q10: Amazon EMR Notebooks '
    f10j = 'Q10:Google Cloud AI Platform Notebooks '
    f10k = 'Q10:Google Cloud Datalab Notebooks'
    f10l = 'Q10: Databricks Collaborative Notebooks '
    f10m = 'Q10:None'
    f10n = 'Q10:Other'
    f14a = 'Q14: Matplotlib '
    f14b = 'Q14: Seaborn '
    f14c = 'Q14: Plotly / Plotly Express '
    f14d = 'Q14: Ggplot / ggplot2 '
    f14e = 'Q14: Shiny '
    f14f = 'Q14: D3 js '
    f14g = 'Q14: Altair '
    f14h = 'Q14: Bokeh '
    f14i = 'Q14: Geoplotlib '
    f14j = 'Q14: Leaflet / Folium '
    f14k = 'Q14:None'
    f14l = 'Q14:Other'
    f16a = 'Q16:  Scikit-learn '
    f16b = 'Q16:  TensorFlow '
    f16c = 'Q16: Keras '
    f16d = 'Q16: PyTorch '
    f16e = 'Q16: Fast.ai '
    f16f = 'Q16: MXNet '
    f16g = 'Q16: Xgboost '
    f16h = 'Q16: LightGBM '
    f16i = 'Q16: CatBoost '
    f16j = 'Q16: Prophet '
    f16k = 'Q16: H2O 3 '
    f16l = 'Q16: Caret '
    f16m = 'Q16: Tidymodels '
    f16n = 'Q16: JAX '
    f16o = 'Q16:None'
    f16p = 'Q16:Other'
    f17a = 'Q17:Linear or Logistic Regression'
    f17b = 'Q17:Decision Trees or Random Forests'
    f17c = 'Q17:Gradient Boosting Machines (xgboost, lightgbm, etc)'
    f17d = 'Q17:Bayesian Approaches'
    f17e = 'Q17:Evolutionary Approaches'
    f17f = 'Q17:Dense Neural Networks (MLPs, etc)'
    f17g = 'Q17:Convolutional Neural Networks'
    f17h = 'Q17:Generative Adversarial Networks'
    f17i = 'Q17:Recurrent Neural Networks'
    f17j = 'Q17:Transformer Networks (BERT gpt-3 etc)'
    f17k = 'Q17:None'
    f17l = 'Q17:Other'
    f23a = 'Q23:Analyze and understand data to influence product or business decisions'
    f23b = 'Q23:Build and/or run the data infrastructure that my business uses for storing analyzing and operationalizing data'
    f23c = 'Q23:Build prototypes to explore applying machine learning to new areas'
    f23d = 'Q23:Build and/or run a machine learning service that operationally improves my product or workflows'
    f23e = 'Q23:Experimentation and iteration to improve existing ML models'
    f23f = 'Q23:Do research that advances the state of the art of machine learning'
    f23g = 'Q23:None of these activities are an important part of my role at work'
    f23h = 'Q23:Other'
    f26a = 'Q26A: Amazon Web Services (AWS) '
    f26b = 'Q26A: Microsoft Azure '
    f26c = 'Q26A: Google Cloud Platform (GCP) '
    f26d = 'Q26A: IBM Cloud / Red Hat '
    f26e = 'Q26A: Oracle Cloud '
    f26f = 'Q26A: SAP Cloud '
    f26g = 'Q26A: Salesforce Cloud '
    f26h = 'Q26A: VMware Cloud '
    f26i = 'Q26A: Alibaba Cloud '
    f26j = 'Q26A: Tencent Cloud '
    f26k = 'Q26A:None'
    f26l = 'Q26A:Other'
    f29a = 'Q29A:MySQL '
    f29b = 'Q29A:PostgresSQL '
    f29c = 'Q29A:SQLite '
    f29d = 'Q29A:Oracle Database '
    f29e = 'Q29A:MongoDB '
    f29f = 'Q29A:Snowflake '
    f29g = 'Q29A:IBM Db2 '
    f29h = 'Q29A:Microsoft SQL Server '
    f29i = 'Q29A:Microsoft Access '
    f29j = 'Q29A:Microsoft Azure Data Lake Storage '
    f29k = 'Q29A:Amazon Redshift '
    f29l = 'Q29A:Amazon Athena '
    f29m = 'Q29A:Amazon DynamoDB '
    f29n = 'Q29A:Google Cloud BigQuery '
    f29o = 'Q29A:Google Cloud SQL '
    f29p = 'Q29A:Google Cloud Firestore '
    f29q = 'Q29A:None'
    f29r = 'Q29A:Other'
    f31a = 'Q31A:Amazon QuickSight'
    f31b = 'Q31A:Microsoft Power BI'
    f31c = 'Q31A:Google Data Studio'
    f31d = 'Q31A:Looker'
    f31e = 'Q31A:Tableau'
    f31f = 'Q31A:Salesforce'
    f31g = 'Q31A:Einstein Analytics'
    f31h = 'Q31A:Qlik'
    f31i = 'Q31A:Domo'
    f31j = 'Q31A:TIBCO Spotfire'
    f31k = 'Q31A:Alteryx '
    f31l = 'Q31A:Sisense '
    f31m = 'Q31A:SAP Analytics Cloud '
    f31n = 'Q31A:None'
    f31o = 'Q31A:Other'
    f33a = 'Q33A:Automated data augmentation (e.g. imgaug, albumentations)'
    f33b = 'Q33A:Automated feature engineering/selection (e.g. tpot, boruta_py)'
    f33c = 'Q33A:Automated model selection (e.g. auto-sklearn, xcessiv)'
    f33d = 'Q33A:Automated model architecture searches (e.g. darts, enas)'
    f33e = 'Q33A:Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)'
    f33f = 'Q33A:Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)'
    f33g = 'Q33A:None'
    f33h = 'Q33A:Other'
    f35a = 'Q35A: Neptune.ai '
    f35b = 'Q35A: Weights & Biases '
    f35c = 'Q35A: Comet.ml '
    f35d = 'Q35A: Sacred + Omniboard '
    f35e = 'Q35A: TensorBoard '
    f35f = 'Q35A: Guild.ai '
    f35g = 'Q35A: Polyaxon '
    f35h = 'Q35A: Trains '
    f35i = 'Q35A: Domino Model Monitor '
    f35j = 'Q35A:None'
    f35k = 'Q35A:Other'

    #définition du questionnaire de questions features
    dict_display_quest = {f6 : "Depuis combien d'années pratiquez-vous le code et/ou la programmation ?",
                          f15 : "Depuis combien d'années utilisez-vous des méthodes de Machine Learning ?",
                          f7a : "Utilisez-vous régulièrement le langage de programmation Python ?",
                          f7b : "Utilisez-vous régulièrement le langage de programmation R ?",
                          f7c : "Utilisez-vous régulièrement le langage de programmation SQL ?",
                          f7d : "Utilisez-vous régulièrement le langage de programmation C ?",
                          f7e : "Utilisez-vous régulièrement le langage de programmation C++ ?",
                          f7f : "Utilisez-vous régulièrement le langage de programmation Java ?",
                          f7g : "Utilisez-vous régulièrement le langage de programmation Javascript ?",
                          f7h : "Utilisez-vous régulièrement le langage de programmation Julia ?",
                          f7i : "Utilisez-vous régulièrement le langage de programmation Swift ?",
                          f7j : "Utilisez-vous régulièrement le langage de programmation Bash ?",
                          f7k : "Utilisez-vous régulièrement le langage de programmation MATLAB ?",
                          f7l : "Vous n'utilisez régulièrement aucun langage de programmation ?",
                          f7m : "Utilisez-vous régulièrement un langage de programmation autre que Python / R / SQL / C / C++ / Java / Javascript / Julia / Swift / Bash / MATLAB ?",
                          f7n : "Utilisez-vous régulièrement l'IDE Jupyter ?",
                          f9a : "Utilisez-vous régulièrement l'IDE RStudio ?",
                          f9b : "Utilisez-vous régulièrement l'IDE Visual Studio ?",
                          f9c : "Utilisez-vous régulièrement l'IDE Visual Studio Code ?",
                          f9d : "Utilisez-vous régulièrement l'IDE PyCharm ?",
                          f9e : "Utilisez-vous régulièrement l'IDE Spyder ? ",
                          f9f : "Utilisez-vous régulièrement l'IDE Notepad++ ?",
                          f9g : "Utilisez-vous régulièrement l'IDE Sublime Text ?",
                          f9h : "Utilisez-vous régulièrement l'IDE Vim / Emacs ?",
                          f9i : "Utilisez-vous régulièrement l'IDE MATLAB ?",
                          f9j : "Vous n'utilisez régulièrement aucun IDE ?",
                          f9k : "Utilisez-vous régulièrement un IDE autre que Jupyter / RStudio / Visual Studio / Visual Studio Code / Pycharm / Spyder / Notepad++ / Sublime Text / Vim/Emacs / MATLAB ?",
                          f10a : "Utilisez-vous régulièrement des notebooks hébergés avec Kaggle Notebooks ?",
                          f10b : "Utilisez-vous régulièrement des notebooks hébergés avec Colab Notebooks ?",
                          f10c : "Utilisez-vous régulièrement des notebooks hébergés avec Azure Notebooks ?",
                          f10d : "Utilisez-vous régulièrement des notebooks hébergés avec Paperspace / Gradient ?",
                          f10e : "Utilisez-vous régulièrement des notebooks hébergés avec Binder / JupyterHub ?",
                          f10f : "Utilisez-vous régulièrement des notebooks hébergés avec Code Ocean ?",
                          f10g : "Utilisez-vous régulièrement des notebooks hébergés avec IBM Watson Studio ?",
                          f10h : "Utilisez-vous régulièrement des notebooks hébergés avec Amazon Sagemaker Studio ?",
                          f10i : "Utilisez-vous régulièrement des notebooks hébergés avec Amazon EMR Notebooks ?",
                          f10j : "Utilisez-vous régulièrement des notebooks hébergés avec Google Cloud AI Platform Notebooks ?",
                          f10k : "Utilisez-vous régulièrement des notebooks hébergés avec Google Cloud Datalab Notebooks ?",
                          f10l : "Utilisez-vous régulièrement des notebooks hébergés avec Databricks Collaborative Notebooks ?",
                          f10m : "Vous n'utilisez régulièrement aucun notebook hébergé ?",
                          f10n : "Utilisez-vous régulièrement des notebooks hébergés autres que Kaggle Notebooks / Colab Notebooks / Azure Notebooks / Paperspace/Gradient / Binder/JupyterHub / Code Ocean / IBM Watson Studio / Amazon Sagemaker Studio / Amazon EMR Notebooks / Google Cloud AI Platform Notebooks / Google Cloud Datalab Notebooks / Databricks Collaborative Notebooks ?",
                          f14a : "Utilisez-vous régulièrement des outils de Data Visualization avec Matplotlib ?",
                          f14b : "Utilisez-vous régulièrement des outils de Data Visualization avec Seaborn ?",
                          f14c : "Utilisez-vous régulièrement des outils de Data Visualization avec Plotly / Plotly Express ?",
                          f14d : "Utilisez-vous régulièrement des outils de Data Visualization avec Ggplot / ggplot2 ?",
                          f14e : "Utilisez-vous régulièrement des outils de Data Visualization avec Shiny ?",
                          f14f : "Utilisez-vous régulièrement des outils de Data Visualization avec D3 js ?",
                          f14g : "Utilisez-vous régulièrement des outils de Data Visualization avec Altair ?",
                          f14h : "Utilisez-vous régulièrement des outils de Data Visualization avec Bokeh ?",
                          f14i : "Utilisez-vous régulièrement des outils de Data Visualization avec Geoplotlib ?",
                          f14j : "Utilisez-vous régulièrement des outils de Data Visualization avec Leaflet / Folium ?",
                          f14k : "Vous n'utilisez régulièrement aucun outil de Data Visualization ?",
                          f14l : "Utilisez-vous régulièrement des outils de Data Visualization autres que Matplotlib / Seaborn / Plotly/Plotly Express / Ggplot/ggplot2 / Shiny / D3 js / Altair / Bokeh / Geoplotlib / Leaflet/Folium ?",
                          f16a : "Utilisez-vous régulièrement des outils de Machine Learning avec Scikit-learn ?",
                          f16b : "Utilisez-vous régulièrement des outils de Machine Learning avec TensorFlow ?",
                          f16c : "Utilisez-vous régulièrement des outils de Machine Learning avec Keras ?",
                          f16d : "Utilisez-vous régulièrement des outils de Machine Learning avec PyTorch ?",
                          f16e : "Utilisez-vous régulièrement des outils de Machine Learning avec Fast.ai ?",
                          f16f : "Utilisez-vous régulièrement des outils de Machine Learning avec MXNet ?",
                          f16g : "Utilisez-vous régulièrement des outils de Machine Learning avec Xgboost ?",
                          f16h : "Utilisez-vous régulièrement des outils de Machine Learning avec LightGBM ?",
                          f16i : "Utilisez-vous régulièrement des outils de Machine Learning avec CatBoost ?",
                          f16j : "Utilisez-vous régulièrement des outils de Machine Learning avec Prophet ?",
                          f16k : "Utilisez-vous régulièrement des outils de Machine Learning avec H2O 3 ?",
                          f16l : "Utilisez-vous régulièrement des outils de Machine Learning avec Caret ?",
                          f16m : "Utilisez-vous régulièrement des outils de Machine Learning avec Tidymodels ?",
                          f16n : "Utilisez-vous régulièrement des outils de Machine Learning avec JAX ?",
                          f16o : "Vous n'utilisez régulièrement aucun outil de Machine Learning ?",
                          f16p : "Utilisez-vous régulièrement des outils de Machine Learning autres que Scikit-learn / TensorFlow / Keras / PyTorch / Fast.ai / MXNet / Xgboost / LightGBM / CatBoost / Prophet / H2O 3 / Caret / Tidymodels / JAX ?" ,
                          f17a : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Linear ou Logistic Regression ?",
                          f17b : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Decision Trees ou Random Forests ?",
                          f17c : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Gradient Boosting Machines (xgboost lightgbm etc) ?",
                          f17d : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Bayesian Approaches ?",
                          f17e : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Evolutionary Approaches ?",
                          f17f : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Dense Neural Networks (MLPs etc) ?",
                          f17g : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Convolutional Neural Networks ?",
                          f17h : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Generative Adversarial Networks ?",
                          f17i : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Recurrent Neural Networks ?",
                          f17j : "Utilisez-vous régulièrement des algorithmes de Machine Learning avec Transformer Networks (BERT gpt-3 etc) ?",
                          f17k : "Vous n'utilisez régulièrement aucun algorithme de Machine Learning ?",
                          f17l : "Utilisez-vous régulièrement des algorithmes de Machine Learning autres que Linear ou Logistic Regression / Decision Trees ou Random Forests / Gradient Boosting Machines (xgboost, lightgbm, etc) / Bayesian Approaches / Evolutionary Approaches / Dense Neural Networks (MLPs, etc) / Convolutional Neural Networks / Generative Adversarial Networks / Recurrent Neural Networks / Transformer Networks (BERT, gpt-3, etc)?",
                          f23a : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Analyser et comprendre la data pour améliorer un produit ou aider aux décisions stratégiques ?",
                          f23b : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Créer et/ou maintenir l'infrastructure de données ?",
                          f23c : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Créer des prototypes pour explorer l'application de Machine Learnings à d'autres domaines ?",
                          f23d : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Créer et/ou maintenir un service de machine learning qui améliore opérationnellement un produit ou des workflows ?",
                          f23e : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Réaliser des expérimentations et des itérations pour améliorer des modèles de machine learning existants ?",
                          f23f : "Est-ce que la tâche suivante constitue une part importante de votre quotidien professionnel: Réaliser des travaux de recherches qui permette d'avancer dans le domaine du Machine Learning ?",
                          f23g : "Le traitement de données et/ou la pratique de Machine Learning ne font pas partie de votre quotidien professionnel ?",
                          f23h : "Le traitement de données et/ou la pratique de Machine Learning ne font pas partie de votre quotidien professionnel ?",
                          f26a : "Utilisez-vous régulièrement la plateforme Amazon Web Services (AWS) ?",
                          f26b : "Utilisez-vous régulièrement la plateforme Microsoft Azure ?",
                          f26c : "Utilisez-vous régulièrement la plateforme Google Cloud Platform (GCP) ?",
                          f26d : "Utilisez-vous régulièrement la plateforme IBM Cloud / Red Hat ?",
                          f26e : "Utilisez-vous régulièrement la plateforme Oracle Cloud ?",
                          f26f : "Utilisez-vous régulièrement la plateforme SAP Cloud ?",
                          f26g : "Utilisez-vous régulièrement la plateforme Salesforce Cloud ?",
                          f26h : "Utilisez-vous régulièrement la plateforme VMware Cloud ?",
                          f26i : "Utilisez-vous régulièrement la plateforme Alibaba Cloud ?",
                          f26j : "Utilisez-vous régulièrement la plateforme Tencent Cloud ?",
                          f26k : "Vous n'utilisez pas régulièrement de plateforme de cloud computing de type Amazon Web Services (AWS) ou Microsoft Azure ?",
                          f26l : "Utilisez-vous régulièrement une plateforme autre que Amazon Web Services (AWS) / Microsoft Azure / Google Cloud Platform (GCP) / IBM Cloud/Red Hat / Oracle Cloud / SAP Cloud / Salesforce Cloud / VMware Cloud / Alibaba Cloud / Tencent Cloud ?",
                          f29a : "Utilisez-vous régulièrement l'outil de Big Data MySQL ?",
                          f29b : "Utilisez-vous régulièrement l'outil de Big Data PostgresSQL ?",
                          f29c : "Utilisez-vous régulièrement l'outil de Big Data SQLite ?",
                          f29d : "Utilisez-vous régulièrement l'outil de Big Data Oracle Database ?",
                          f29e : "Utilisez-vous régulièrement l'outil de Big Data MongoDB ?",
                          f29f : "Utilisez-vous régulièrement l'outil de Big Data Snowflake ?",
                          f29g : "Utilisez-vous régulièrement l'outil de Big Data IBM Db2 ?",
                          f29h : "Utilisez-vous régulièrement l'outil de Big Data Microsoft SQL Server ?",
                          f29i : "Utilisez-vous régulièrement l'outil de Big Data Microsoft Access ?",
                          f29j : "Utilisez-vous régulièrement l'outil de Big Data Microsoft Azure Data Lake Storage ?",
                          f29k : "Utilisez-vous régulièrement l'outil de Big Data Amazon Redshift ?",
                          f29l : "Utilisez-vous régulièrement l'outil de Big Data Amazon Athena ?",
                          f29m : "Utilisez-vous régulièrement l'outil de Big Data Amazon DynamoDB ?",
                          f29n : "Utilisez-vous régulièrement l'outil de Big Data Google Cloud BigQuery ?",
                          f29o : "Utilisez-vous régulièrement l'outil de Big Data Google Cloud SQL ?",
                          f29p : "Utilisez-vous régulièrement l'outil de Big Data Google Cloud Firestore ?",
                          f29q : "Vous n'utilisez pas régulièrement d'outil de Big Data ?",
                          f29r : "Utilisez-vous régulièrement un outil de Big Data autre que MySQL / PostgreSQL / SQLite / Oracle Database / MongoDB / Snowflake / IBM Db2 / Microsoft SQL Server / Microsoft Access / Microsoft Azure Data Lake Storage / Amazon Redshift / Amazon Athena / Amazon DynamoDB / Google Cloud BigQuery / Google Cloud SQL / Google Cloud Firestore ?",
                          f31a : "Utilisez-vous régulièrement l'outil de business intelligence Amazon QuickSight ?",
                          f31b : "Utilisez-vous régulièrement l'outil de business intelligence Microsoft Power BI ?",
                          f31c : "Utilisez-vous régulièrement l'outil de business intelligence Google Data Studio ?",
                          f31d : "Utilisez-vous régulièrement l'outil de business intelligence Looker ?",
                          f31e : "Utilisez-vous régulièrement l'outil de business intelligence Tableau ?",
                          f31f : "Utilisez-vous régulièrement l'outil de business intelligence Salesforce ?",
                          f31g : "Utilisez-vous régulièrement l'outil de business intelligence Einstein Analytics ?",
                          f31h : "Utilisez-vous régulièrement l'outil de business intelligence Qlik ?",
                          f31i : "Utilisez-vous régulièrement l'outil de business intelligence Domo ?",
                          f31j : "Utilisez-vous régulièrement l'outil de business intelligence TIBCO Spotfire ?",
                          f31k : "Utilisez-vous régulièrement l'outil de business intelligence Alteryx ?",
                          f31l : "Utilisez-vous régulièrement l'outil de business intelligence Sisense ?",
                          f31m : "Utilisez-vous régulièrement l'outil de business intelligence SAP Analytics Cloud ?",
                          f31n : "Vous n'utilisez pas régulièrement d'outil de business intelligence ?",
                          f31o : "Utilisez-vous régulièrement un outil de business intelligence autre que Amazon QuickSight / Microsoft Power BI / Google Data Studio / Looker / Tableau / Salesforce / Einstein Analytics / Qlik / Domo / TIBCO Spotfire / Alteryx / Sisense / SAP Analytics Cloud ?",
                          f33a : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automated data augmentation (e.g. imgaug, albumentations) ?",
                          f33b : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automated feature engineering/selection (e.g. tpot, boruta_py) ?",
                          f33c : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automated model selection (e.g. auto-sklearn, xcessiv) ?",
                          f33d : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automated model architecture searches (e.g. darts, enas) ?",
                          f33e : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier) ?",
                          f33f : "Utilisez-vous régulièrement des outils de Machine Learning automatisés de type Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI) ?",
                          f33g : "Vous n'utilisez pas régulièrement des outils de Machine Learning automatisés ?",
                          f33h : "Vous n'utilisez pas régulièrement des outils de Machine Learning automatisés ?",
                          f35a : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Neptune.ai ?",
                          f35b : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Weights & Biases ?",
                          f35c : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Comet.ml ?",
                          f35d : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Sacred + Omniboard ?",
                          f35e : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning TensorBoard ?",
                          f35f : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Guild.ai ?",
                          f35g : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Polyaxon ?",
                          f35h : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Trains ?",
                          f35i : "Utilisez-vous régulièrement l'outil d'aide à la gestion des expérimentations de machine learning Domino Model Monitor ?",
                          f35j : "Vous n'utilisez pas régulièrement d'outil d'aide à la gestion des expérimentations de machine learning ?",
                          f35k : "Vous n'utilisez pas régulièrement d'outil d'aide à la gestion des expérimentations de machine learning ?"}


    #Liste des réponses et dictionnaires associés
    #*********************************************
    
    r6 = ["Je n'ai jamais codé","Moins d'1 an", "Entre 1 et 2 ans", "Entre 3 et 5 ans", "Entre 5 et 10 ans", "Entre 10 et 20 ans", "Plus de 20 ans"]
    r15 = ["Je n'ai jamais utilisé de méthodes de Machine Learning", "Moins d'1 an", "Entre 1 et 2 ans", "Entre 2 et 3 ans",
           "Entre 3 et 4 ans", "Entre 4 et 5 ans", "Entre 5 et 10 ans", "Entre 10 et 20 ans", "Plus de 20 ans"]
    rOther=['Oui', 'Non']


    dico_6 = {"Je n'ai jamais codé" : 0,
              "Moins d'1 an" : 1,
              "Entre 1 et 2 ans" : 2,
              "Entre 3 et 5 ans" : 3,
              "Entre 5 et 10 ans" : 4,
              "Entre 10 et 20 ans" : 5,
              "Plus de 20 ans" : 6}

    dico_15 = {"Je n'ai jamais utilisé de méthodes de Machine Learning" : 0,
               "Moins d'1 an" : 1,
               "Entre 1 et 2 ans" : 2,
               "Entre 2 et 3 ans" : 3,
               "Entre 3 et 4 ans" : 4,
               "Entre 4 et 5 ans" : 5,
               "Entre 5 et 10 ans" : 6,
               "Entre 10 et 20 ans" : 7,
               "Plus de 20 ans" : 8}

    dico_bin = {"Oui" : 1, "Non" : 0}


    #Création du vecteur 'target' contenant la variable cible 'Q5' et d'un Data Frame 'feats' contenant les différentes features.
    target = df_clean['Q5']
    feats=df_clean.drop('Q5', axis=1)

        
    with demo:

        st.header('Démo')
        st.header('')
        st.markdown("Pour simplifier cette démonstration et permettre un calcul rapide, le modèle utilisé est une **Régression Logistique** avec une réduction de dimension via SelectKBest, k par défaut égal à 10 et score_func égal à mutual_info_classif.")
        st.markdown("Cela a pour conséquence de dégrader de 3 à 5 points la performance comparée aux résultats des modèles optimisés précédemment.")

        st.subheader('Testons le modèle')

        Formulaire = st.checkbox("Veuillez remplir le formulaire ci-dessous",False,key='Form1')
        st.caption("Cochez la case pour afficher le questionnaire")
        
        if Formulaire == 1:
      
            col1, col2, col3 = st.columns(3)

            with col2:
                attente = st.image("WorkInProgress.jpg") 
    
            #Séparation du dataset en train set et test set
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=200)
            #Standardisation
            sc= StandardScaler()
            X_train_scaled = sc.fit_transform(X_train)
            X_test_scaled = sc.transform(X_test)

            lr = LogisticRegression(random_state=200)
            sel = SelectKBest(score_func = mutual_info_classif)
            sel.fit(X_train_scaled, y_train)

            attente.empty()
        
            #Transformation des train set et test set
            sel_train = sel.transform(X_train_scaled)
            sel_test = sel.transform(X_test_scaled)
            #Prédiction sur sel_test
            lr.fit(sel_train, y_train)

            st.write("Score du train set:", round(lr.score(sel_train,y_train)*100,2),'%')
            st.write("Score du test set:", round(lr.score(sel_test,y_test)*100,2),'%')
            st.header("")
        
            
            #Questionnaire
            #**************
        
            x0 = feats.columns[sel.get_support()][0]
            x1 = feats.columns[sel.get_support()][1]
            x2 = feats.columns[sel.get_support()][2]
            x3 = feats.columns[sel.get_support()][3]
            x4 = feats.columns[sel.get_support()][4]
            x5 = feats.columns[sel.get_support()][5]
            x6 = feats.columns[sel.get_support()][6]
            x7 = feats.columns[sel.get_support()][7]
            x8 = feats.columns[sel.get_support()][8]
            x9 = feats.columns[sel.get_support()][9]
        
            with st.form("my_form"):
            
                st.subheader('Les 10 questions retenues par le modèle sont :\n')
                st.header('\n')
    
                #Question 1
                #***********
                st.write("#01 - ",dict_display_quest[x0])
            
                if x0 == f6:
                    a0 = st.radio("",r6)
                    b0 = dico_6[a0]
                elif x0 == f15:
                    a0 = st.radio("",r15)
                    b0 = dico_15[a0]
                else:
                    a0 = st.radio("",rOther,index=1,key=0)
                    b0 = dico_bin[a0]

                #Question 2
                #***********
                st.header("")
                st.write("#02 - ",dict_display_quest[x1])
                if x1 == f6:
                    a1 = st.radio("",r6)
                    b1 = dico_6[a1]
                elif x1 == f15:
                    a1 = st.radio("",r15)
                    b1 = dico_15[a1]
                else:
                    a1 = st.radio("",rOther,index=1,key=1)
                    b1 = dico_bin[a1]
                    
                
                #Question 3
                #***********
                st.header("")
                st.write("#03 - ",dict_display_quest[x2])
                if x2 == f6:
                    a2 = st.radio("",r6)
                    b2 = dico_6[a2]
                elif x2 == f15:
                    a2 = st.radio("",r15)
                    b2 = dico_15[a2]
                else:
                    a2 = st.radio("",rOther,index=1,key=2)
                    b2 = dico_bin[a2]
                    
                
                #Question 4
                #***********
                st.header("")
                st.write("#04 - ",dict_display_quest[x3])
                st.markdown("")
                if x3 == f6:
                    a3 = st.radio("",r6)
                    b3 = dico_6[a3]
                elif x3 == f15:
                    a3 = st.radio("",r15)
                    b3 = dico_15[a3]
                else:
                    a3 = st.radio("",rOther,index=1,key=3)
                    b3 = dico_bin[a3]
                
                
                #Question 5
                #***********
                st.header("")
                st.write("#05 - ",dict_display_quest[x4])
                if x4 == f6:
                    a4 = st.radio("",r6)
                    b4 = dico_6[a4]
                elif x4 == f15:
                    a4 = st.radio("",r15)
                    b4 = dico_15[a4]
                else:
                    a4 = st.radio("",rOther,index=1,key=4)
                    b4 = dico_bin[a4]
                
                
                #Question 6
                #***********
                st.header("")
                st.write("#06 - ",dict_display_quest[x5])
                if x5 == f6:
                    a5 = st.radio("",r6)
                    b5 = dico_6[a5]
                elif x5 == f15:
                    a5 = st.radio("",r15)
                    b5 = dico_15[a5]
                else:
                    a5 = st.radio("",rOther,index=1,key=5)
                    b5 = dico_bin[a5]
                
                
                #Question 7
                #***********
                st.header("")
                st.write("#07 - ",dict_display_quest[x6])
                if x6 == f6:
                    a6 = st.radio("",r6)
                    b6 = dico_6[a6]
                elif x6 == f15:
                    a6 = st.radio("",r15)
                    b6 = dico_15[a6]
                else:
                    a6 = st.radio("",rOther,index=1,key=6)
                    b6 = dico_bin[a6]
                
                
                #Question 8
                #***********
                st.header("")
                st.write("#08 - ",dict_display_quest[x7])
                if x7 == f6:
                    a7 = st.radio("",r6)
                    b7 = dico_6[a7]
                elif x7 == f15:
                    a7 = st.radio("",r15)
                    b7 = dico_15[a7]
                else:
                    a7 = st.radio("",rOther,index=1,key=7)
                    b7 = dico_bin[a7]
              
                
                #Question 0
                #***********
                st.header("")
                st.write("#09 - ",dict_display_quest[x8])
                if x8 == f6:
                    a8 = st.radio("",r6)
                    b8 = dico_6[a8]
                elif x8 == f15:
                    a8 = st.radio("",r15)
                    b8 = dico_15[a8]
                else:
                    a8 = st.radio("",rOther,index=1,key=8)
                    b8 = dico_bin[a8]
                
                
                #Question 10
                #***********
                st.header("")
                st.write("#10 - ",dict_display_quest[x9])
                if x9 == f6:
                    a9 = st.radio("",r6)
                    b9 = dico_6[a9]
                elif x9 == f15:
                    a9 = st.radio("",r15)
                    b9 = dico_15[a9]
                else:
                    a9 = st.radio("",rOther,index=1,key=9)
                    b9 = dico_bin[a9]


                st.header("")
        

                # Every form must have a submit button.
                submitted = st.form_submit_button("Valider le questionnaire")
                processed = st.form_submit_button("Actionner la modélisation")
                
                if submitted:
                    st.success("Merci. Votre formulaire a été pris en compte.")

                    with st.expander("Réponses fournies:"):
                        Data = {'Question':[x0,x1,x2,x3,x4,x5,x6,x7,x8,x9],
                                'Réponse':[a0,a1,a2,a3,a4,a5,a6,a7,a8,a9],
                                'Code':[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9]}
                        dfResults = pd.DataFrame(Data)
                        st.dataframe(dfResults)
                        
                        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=200)
                        sc= StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        lr = LogisticRegression(random_state=200)
                        sel = SelectKBest(score_func = mutual_info_classif)
                        sel.fit(X_train_scaled, y_train)

                        sel_train = sel.transform(X_train_scaled)
                        sel_test = sel.transform(X_test_scaled)
                        lr.fit(sel_train, y_train)

                        #Création du tableau de réponses
                        x_pred_init = {feats.columns[sel.get_support()].tolist()[0] : [b0],
                                       feats.columns[sel.get_support()].tolist()[1] : [b1],
                                       feats.columns[sel.get_support()].tolist()[2] : [b2],
                                       feats.columns[sel.get_support()].tolist()[3] : [b3],
                                       feats.columns[sel.get_support()].tolist()[4] : [b4],
                                       feats.columns[sel.get_support()].tolist()[5] : [b5],
                                       feats.columns[sel.get_support()].tolist()[6] : [b6],
                                       feats.columns[sel.get_support()].tolist()[7] : [b7],
                                       feats.columns[sel.get_support()].tolist()[8] : [b8],
                                       feats.columns[sel.get_support()].tolist()[9] : [b9],}

                        x_pred = pd.DataFrame(x_pred_init)
                        st.dataframe(x_pred)

                        reponse = lr.predict(x_pred).tolist()[0]
                        st.markdown("")
                        st.markdown("")
                        st.markdown("Vous semblez avoir un profil de :")
                        st.header(reponse)
                    
                    Formulaire2 = st.checkbox("Votre nouvelle carrière vous ouvre les bras :",False,key='Form2')
                    st.caption("Cochez la case, puis cliquez sur 'Actionner la modélisation' pour afficher quel poste correspond à votre profil.")
                
                if processed:
                
                    Formulaire2 = st.checkbox("Votre nouvelle carrière vous ouvre les bras :",False,key='Form2')
                    st.caption("Cochez la case, puis cliquez sur 'Actionner la modélisation' pour afficher quel poste correspond à votre profil.")
                    
                    if Formulaire2 == 1:
                        
                        attente2 = st.image("WorkInProgress.jpg") 
                        df_clean = pd.read_csv('df_clean.csv')
                        col1,col2 = st.columns([3,1])
                        
                        #rappel du modèle de régression logistique ==> colonne1
                        #*******************************************************
                        x_pred_init = {feats.columns[sel.get_support()].tolist()[0] : [b0],
                                       feats.columns[sel.get_support()].tolist()[1] : [b1],
                                       feats.columns[sel.get_support()].tolist()[2] : [b2],
                                       feats.columns[sel.get_support()].tolist()[3] : [b3],
                                       feats.columns[sel.get_support()].tolist()[4] : [b4],
                                       feats.columns[sel.get_support()].tolist()[5] : [b5],
                                       feats.columns[sel.get_support()].tolist()[6] : [b6],
                                       feats.columns[sel.get_support()].tolist()[7] : [b7],
                                       feats.columns[sel.get_support()].tolist()[8] : [b8],
                                       feats.columns[sel.get_support()].tolist()[9] : [b9],}
                        
                        x_pred = pd.DataFrame(x_pred_init)
                        reponse = lr.predict(x_pred).tolist()[0]
                        
                        #définition des paramètres du graphique ==> colonne2
                        #*******************************************************
                        comptage = df_clean['Q5'].value_counts()
                        values = comptage.tolist()
                        names = comptage.index.tolist()
                        print(names)

                        pos_color = names.index(reponse)
                        
                        css_color = []
                        
                        for i in range(5):
                            if i == pos_color:
                                css_color.append('#90EE90')
                            else:
                                css_color.append('#FFFACD')
                                
                        
                        fig = px.pie(df_clean, values = values, names = names,
                                     color_discrete_sequence = css_color,
                                     ) 
                        fig.update_traces(textinfo= 'percent+label',
                                          showlegend = False,
                                          hovertemplate = "<b>%{label}:</b> <br> <i>%{value}</i> </br> %{percent:.2%f}",
                                          texttemplate='<b>%{label}:</b> <br> %{percent:.2%f}',
                                          hole = 0.2,
                                          )
                        fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
                        attente2.empty()
                        

                        with col1:
                            st.write(fig)
                                               
                        with col2:
                            st.markdown("Vous êtes sur le point de démarrer une carrière de :")
                            st.header(reponse)
                            st.header(":thumbsup:")
                    
                    
#*******************************            


#*******************************
# Mise en page de la CONCLUSION
#*******************************

if option == 'Conclusion':
        
    with conclusion:
        st.header("Conclusion")
        st.text("")
        st.markdown("Notre étude des différents métiers de la data nous a conduit à analyser un jeu de données représentant l’ensemble des réponses de plus de 20 000 individus à une quarantaine de questions, ceci dans le but d’élaborer un modèle de prédiction de postes data en fonction des compétences des individus.")
        st.markdown("Après exclusion des non-professionnels (étudiants, sans-emploi et autres), le panel de professions classifiées a été réduit de 10 à 5 (soit 80% du panel de répondants) afin de ne pas surcharger la modélisation.")
        st.markdown("Cinq modèles ont été retenus : Logistic Regression, SVM, Random Forest, Decision Tree et KNN , produisant des prédictions dont la précision (score test) variait entre 38 et 54%.")
        st.markdown("Une faible corrélation des variables entre elles ainsi qu’un sur apprentissage des modèles nous ont conduit à explorer différentes techniques (réduction de dimensions, ré-échantillonnage, optimisation des hyper paramètres) dans le but d’améliorer la performance globale des modèles. ")
        st.markdown("Cependant nos différents tests ne nous ont pas permis d’observer une amélioration notable des scores aussi bien dans le sur apprentissage que dans la précision globale. ")
        st.markdown("Plusieurs hypothèses peuvent rentrer en ligne de compte pour expliquer ces scores : ")
        st.markdown("• Le domaine de la Data Science est encore récent et en évolution, ne permettant pas une distinction nette de ses postes au vu des compétences et tâches quotidiennes. ")
        st.markdown("• Des besoins de polyvalence et d’interchangeabilité peuvent aussi être recherchés par les entreprises, soit dans un but évolutif ou tout simplement car la taille de l’entreprise ne permet pas de recruter des spécialistes de chaque métier. ")
        st.markdown("• On ne peut aussi exclure que le questionnaire, bien que fourni, aurait pu être enrichi de questions complémentaires qui auraient accentué les distinctions potentielles entre chaque métier. ")
        st.markdown("Toutefois, nous retenons le modèle **Support Vector Machine** qui présente les meilleures performances relatives, avec pour paramètre kernel = 'linear' et une réduction de dimensions via SelectFromModel. ")
        st.markdown("Il serait intéressant d’inclure les features sélectionnées par SVM dans un nouveau questionnaire que l’on soumettrait à l’échantillon de non professionnels, non retenus durant la modélisation. On ne retiendrait que les individus ayant répondu à toutes ces questions / features. Quels résultats obtiendrions-nous alors ?")

        st.header("Perspectives")
        st.text("")
        st.markdown("L’application présentée dans ce projet reste relativement simple et nécessiterait des développements avant d’être déployée.")
        st.markdown("Il est tout d’abord possible qu’il y ait des biais que nous n’avons pas relevés, par exemple le déséquilibre de répartition des nationalités parmi les répondants. C‘est un peu la problématique de représentativité propre aux sondages.")
        st.markdown("On pourrait ré intégrer certaines questions (features) que nous avions mises de côté. Principalement les questions à tiroirs qui s’adressent peut-être à des métiers plus spécialisés. Mais aussi l’âge pour pouvoir dissocier la prédiction pour les juniors et les seniors.")
        st.markdown("Parmi les modèles retenus, il y a probablement d’autres tests et optimisation d’hyperparamètres à essayer.  D’autres modèles pourraient aussi être testés, peut-être GradientBoosting, XGBoost ou autres.")
        st.markdown("Ce questionnaire pourrait être adapté à des marchés nationaux (définitions de postes, salaires et expérience) pour affiner la performance du modèle. Une fois fait, notre fichier démo pourrait alors servir aux cabinets de recrutement tech et entreprises spécialisées dans la data, qui soumettraient le questionnaire développé aux candidats potentiels.")
        
            
#*******************************


            