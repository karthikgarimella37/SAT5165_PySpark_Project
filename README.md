# Spark for Big Data Processing and Statistical Analysis

Implementation of machine learning algorithms to predict the average temperature for each state in the United States over time, using PySpark.

##### Machine Learning Algorithms used:
- Random Forest Regression
- Gradient Boosted Tree Regression
- Decision Tree Regression
- Elastic Net Linear Regression

The preprocessed data of 10,000 records was put through a pipeline of ML algorithms. Using metrics like Rsquared and RMSE, an appropriate model was determined. The entire task was performed in parallel using PySpark and different combinations of virtual machines. The computation speeds for each combination of VMs were recorded.