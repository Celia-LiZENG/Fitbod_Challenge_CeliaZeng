# Data Recipe Guide from Celia 😋

# Part0: Data Explanation
(1) fitbud_rawData.csv: the assignment has given, which is the original data

(2) ob_perform.csv: data transformed from original fitbud_rawData, using pipeline functions from EDA part. In fact, any workout histroy data can be transformed into this one. This can be seen as part of users' fitness portrait data (transformed from bahavior data).

# Part1: EDA Analysis (Y: 1 if churn, 0 if stay at Fitbod)
(1) Since i have no data description about columns, I will firstly understand the data columns with some simple analysis and my usage of Fitbod Apll;

(2) summary statistics, daily/montly active user analysis; 

(3) the motivation to build a churn model to predict users' churn rate in the following time peried: new users acquired slowly in Covid-19 period and important to retain the original old users;

(4) see how input attributes related to churn?

(5) Inspirations from EDA to build the ML model

# Part2: Build Churn Machine Learning Models
(1) Choose 3 algorithms (can try more but here I only pick 3)

(2) Use grid search to find the best parameters

(3) Performance comparison

# Part3: Display ML Models and Visulizations via Streamlit
  Here I used Streamlit to display my model. This is an interaction web which was simply based on the re-written Python code from Part2. I prefer to use this one as it can show how the churn prediction will change when different data attributes change.
  
  🔥 Application: Product analysts and data scientist in Fitbod in can easily use it to get the churn prediction for particular user and also compare different models!
  
# Part4: Limitations
  In the later presentation, I will talk about my EDA analysis, ML models using PPT and Streamlit! I will also talk about how we can extend this data to gain more insights and achieve more functions. It is my great pelasure to explore more and study in Fitbod!

