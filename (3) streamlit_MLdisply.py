#-----<Step1: Intstall Required Packages>---------
import streamlit as st
import pickle
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score

# Title and Subheader
from PIL import Image
image = Image.open('fitbod_logo2.jpg')
st.image(image, width=1,caption=None,
    use_column_width=True)
st.title("Fitbod Helper: Predict Churn User")
st.header("Step0: Understand Dataset")

image = Image.open('new_churn.jpg')
st.image(image, width=1,caption='New User VS. Churn User',
    use_column_width=True)


# Load the pipeline and data
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))
workoutData = pickle.load(open('workoutData.sav', 'rb'))
replace_cols=['churn_user']
for i in replace_cols : 
    y_test[i]  = y_test[i].replace({1.0 : 1, 0.0:0})

pipe1 = pickle.load(open('pipe_KNN.sav', 'rb'))
knn_pre_class = pipe1.predict(X_test)
knn_pre_prob = pipe1.predict_proba(X_test)
#score1 = pipe1.score(X_test, y_test)
score1 = 0.688
#auc1 = cross_val_score(pipe1, X_test, y_test, scoring='roc_auc', cv=10)
# precision_KNN= precision_score(y_true=y_test, y_pred=knn_pre_class)
precision_KNN = 0.500
# recall_KNN=recall_score(y_true=y_test, y_pred=knn_pre_class)
recall_KNN = 0.200
#f1_KNN=f1_score(y_true=y_test, y_pred=knn_pre_class)
f1_KNN = 0.286


pipe2 = pickle.load(open('pipe_RF.sav', 'rb'))
rf_pre_class = pipe2.predict(X_test)
rf_pre_prob = pipe2.predict_proba(X_test)
#score2 = pipe2.score(X_test, y_test)
score2=0.938
#auc1 = cross_val_score(pipe1, X_test, y_test, scoring='roc_auc', cv=10)
#precision_RF= precision_score(y_true=y_test, y_pred=rf_pre_class)
precision_RF=0.500
#recall_RF=recall_score(y_true=y_test, y_pred=rf_pre_class)
recall_RF= 0.400
#f1_RF=f1_score(y_true=y_test, y_pred=rf_pre_class)
f1_RF=0.444
    
pipe3 = pickle.load(open('pipe_LR.sav', 'rb'))
lr_pre_class = pipe3.predict(X_test)
lr_pre_prob = pipe3.predict_proba(X_test)
#score3 = pipe3.score(X_test, y_test)
score3=0.625
#auc1 = cross_val_score(pipe1, X_test, y_test, scoring='roc_auc', cv=10)
#precision_LR= precision_score(y_true=y_test, y_pred=lr_pre_class)
precision_LR=0.400
#recall_LR=recall_score(y_true=y_test, y_pred=lr_pre_class)
recall_LR = 0.400
#f1_LR=f1_score(y_true=y_test, y_pred=lr_pre_class)
f1_LR = 0.400

clf_compare ={'Accuracy':[score1,score2,score3],
                     'precision':[precision_KNN,precision_RF,precision_LR],
                     'recall':[recall_KNN,recall_RF,recall_LR],
                     'F1 Score':[f1_KNN,f1_RF,f1_LR]}
clf_compare = pd.DataFrame(clf_compare)
clf_compare.index=['KNN','Random Forest','Logistic Regression']

if st.checkbox('Show User Workout History Data (original)--->>>'):
    st.write(workoutData.head())
uploaded_file = st.file_uploader("Choose a CSV file in the Workout History Format", type="csv")
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

if st.checkbox('Show Processed Data (Model Inputs)--->>>'):
    st.write(X_test.head())

def test_demo(index):
    values = X_test.iloc[index] 
    image = Image.open('user3.png')
    st.sidebar.image(image, width=1,caption='User Workout History Portrait',
    use_column_width=True)
    # Create User Input Data in the sidebar/text
    active_days = st.sidebar.slider("How Many Days User Has Workouted with Fitbod?")
    workout_times = st.sidebar.slider("What's the Total Workout Times?")
    workout_per_day = st.sidebar.slider("What's the Workout Times per Day?")
    volume_total = st.sidebar.text_input("What's the Total Exercise Volume on Fitbod?")
    volume_per_day = st.sidebar.slider("Volume(Reps) per Day?")
    intensity_total = st.sidebar.text_input("Total Intensity(Weight)?")
    intensity_per_day = st.sidebar.slider("Intensity(Weight) per Day?")
    favorite_exercise_0 = st.sidebar.selectbox("User's Most Frequently Chosen Exercise?", workoutData.exerciseName.unique().tolist(),key='2')
    dic = {'Barbell Bench Press':0.22, 'Dumbbell Bench Press':0.33, 'Dumbbell Bench Press':0.307, 'Close-Grip Bench Press':0.57,
    'Deadlift':0.3077, 'Dumbbell Lunge':0.4286, 'Dumbbell Shoulder Press':0.50, 'Dumbbell Row':0.6, 'Dumbbell Bent Over Row':0.4
    }
    favorite_exercise= dic[favorite_exercise_0]
    most_recent_exercise_0 = st.sidebar.selectbox("User's Latest Chosen Exercise?",workoutData.exerciseName.unique().tolist(),key='2')
    most_recent_exercise=dic[most_recent_exercise_0]
    
    stay_days = st.sidebar.slider("Total Days User Has Registered on Fitbod?")
    active_extent= st.sidebar.slider("User's Active Extent?",min_value=0.00, max_value=1.00, step=0.01)
    last_to_now= st.sidebar.slider("How Many Days Went since User's Latest Workout?")
    high_volume_user = st.sidebar.slider("Whether User is High Volume?")
    high_intensity_user = st.sidebar.slider("Whether User is High Intensity User?")
    high_active_user = st.sidebar.slider("Whether User is Highly Active?")
    
    
    #Print the prediction result
    st.balloons()
    st.header("Step1: Choose Model")
    if st.checkbox('See the model performance comparison before choosing your favorite one :)'):
        st.write(clf_compare)
        st.write('               🎉 Random Forest Wins the Best AUC Score！！    ')
        image = Image.open('model_AUC.jpg')
        st.image(image, width=1,caption='AUC comparison of 3 Trained Models (Examples)',
    use_column_width=True)
    alg = ['KNN', 'Random Forest', 'Logistic Regression']
    classifier = st.selectbox('Which model do you want to choose for prediction?', alg)
    st.header("Step2: See Machine Learning prediction for this user!")
    dic2={1:'Attention! This is potential churn user!',0:'Congras! This user will stay in Fitbod in the prediction period!'}
    
    if classifier == 'KNN':
        #pip1 = pickle.load(open('pipe_KNN.sav', 'rb'))
        user_prediction_data = np.array([active_days,
       workout_times, workout_per_day, volume_total, volume_per_day,
       intensity_total, intensity_per_day, favorite_exercise,
       most_recent_exercise, stay_days, active_extent, last_to_now,
       high_volume_user, high_intensity_user, high_active_user]).reshape(1,15) 
        
        res = pipe1.predict_proba(user_prediction_data)
        churn = pipe1.predict(user_prediction_data)
        st.write('According to your input, the churn rate is predicted to be', res[:,1])
        if churn ==0:
            st.write('🎉Congras! Our model suggests that this user will stay in Fitbod for the prediction period!')
        elif churn==1:
            st.write('😢Attention! This is a potential churn user!')
        st.text('Notice: the recall of this model means',)
        image = Image.open('knn_recall.jpg')
        st.image(image, width=1,
    use_column_width=True)
        #st.write('Recall: ', recall_KNN)
        
        #st.write('Confusion Matrix: ', cm)
        
    
    elif classifier == 'Random Forest':
        #pip2 = pickle.load(open('pipe_RF.sav', 'rb'))
        user_prediction_data = np.array([active_days,
       workout_times, workout_per_day, volume_total, volume_per_day,
       intensity_total, intensity_per_day, favorite_exercise,
       most_recent_exercise, stay_days, active_extent, last_to_now,
       high_volume_user, high_intensity_user, high_active_user]).reshape(1,15) 
       
        res = pipe2.predict_proba(user_prediction_data)
        churn = pipe2.predict(user_prediction_data)
        st.write('According to your input, the churn rate is predicted to be', res[:,1])
        if churn ==0:
            st.write('🎉Congras! Our model suggests that this user will stay in Fitbod for the prediction period!')
        elif churn==1:
            st.write('😢Attention! This is a potential churn user!')
        st.text('Notice: the recall of this model means',)
        image = Image.open('rf_lr_recall.jpg')
        st.image(image, width=1,
    use_column_width=True)

        image = Image.open('rf_feature.jpg')
        st.image(image, width=1,caption='Feature Importance for Random Forest',
    use_column_width=True)
       
        #st.write('Confusion Matrix: ', cm)
        
        
    elif classifier == 'Logistic Regression':
        #if st.checkbox('See the interpretation of model--->>>'):
            #image = Image.open('LR_feature_importance.jpg')
            #st.image(image, width=1,caption=None, use_column_width=True)
        user_prediction_data = np.array([active_days,
       workout_times, workout_per_day, volume_total, volume_per_day,
       intensity_total, intensity_per_day, favorite_exercise,
       most_recent_exercise, stay_days, active_extent, last_to_now,
       high_volume_user, high_intensity_user, high_active_user]).reshape(1,15) 
        
        res = pipe3.predict_proba(user_prediction_data)
        churn = pipe3.predict(user_prediction_data)
        st.write('According to your input, the churn rate is predicted to be', res[:,1])
        if churn ==0:
            st.write('🎉Congras! Our model suggests that this user will stay in Fitbod for the prediction period!')
        elif churn==1:
            st.write('😢Attention! This is a potential churn user!')
        st.text('Notice: the recall of this model means',)
        image = Image.open('rf_lr_recall.jpg')
        st.image(image, width=1,
    use_column_width=True)
        
        image = Image.open('LR_feature.png')
        st.image(image, width=1,caption='Feature Importance for Logistic Regression',
    use_column_width=True)
        #st.write('Confusion Matrix: ', cm)
number = st.text_input('Choose sample from dataset (0 - 200):', 0)  # Input the index number       
test_demo(int(number))  # Run the test function