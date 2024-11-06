# Importing Libraries
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, substring, to_date, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.pandas import get_dummies
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, DecisionTreeRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from scipy.stats import boxcox


# Setting up PyArrow timezone
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


# Creating Spark Session
spark = SparkSession.builder \
        .appName("Spark Processing").getOrCreate()

# Reading csv file using Spark
df_read = spark.read.csv("file:///home/sat3812/Downloads/US_Weather_Data_2015_2023.csv", header = True, inferSchema=True)

# Printing dataframe to check dataframe data
print(df_read.limit(10))

# Limiting data size 
df = df_read.limit(10000)

# Extracting date, month, year from date column
df = df.withColumn("year", substring(col("date"), 1, 4)) \
	.withColumn("month", substring(col("date"), 5, 2)) \
	.withColumn("day", substring(col("date"), 7, 2))

predictor = df.tavg
# Dropping unused columns
df = df.drop("date", "stability")

# Converting to pandas df on spark
df_ps = df.to_pandas_on_spark()

# Creating dummies for st_abb i.e. State Abbreviations
df_encoded = get_dummies(df_ps, columns = ["st_abb"])

df_encoded = df_encoded.drop(columns = ["county_name"])

# Checking data and columns created
print(df_encoded.head(10))
print(df_encoded.columns)

# Converting all columns to float
for column in df_encoded.columns:
    try:
        df_encoded[column] = df_encoded[column].astype(float)  # Ensure conversion to float
    except Exception as e:
        print(f"Could not convert column {column}: {e}")


df_encoded = df_encoded.select_dtypes(include=['float64'])

# Converting pandas dataframe to pyspark dataframe
df_encoded_spark = df_encoded.to_spark()

feature_columns = [col for col in df_encoded_spark.columns if col != "tavg"] # [col for col in df_encoded_spark.columns]  # Extracting column names as a list

# Print to check the column types
print(df_encoded_spark.dtypes)


# Center and Scaling with StandardScaler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_vector = assembler.transform(df_encoded_spark)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_vector)
scaled_data = scaler_model.transform(df_vector)
scaled_data = scaled_data.withColumn("features_array", vector_to_array("scaled_features"))


normalized_columns = [col for col in df_encoded.columns if col != "tavg"]

df_normalized = scaled_data.select(
    *[F.col("features_array")[i].alias(name) for i, name in enumerate(normalized_columns)]
)

# Converting pyspark df to pandas df
scaled_df_ps = df_normalized.to_pandas_on_spark()

df_normalized.show(truncate=False)


## Extra preprocessing steps if necessary----------------------------------------------------

# Calculating Correlation between columns

feature_columns = [col for col in df_normalized.columns if col != "tavg"] #df_normalized.columns 

# Creating a vectorized feature
vector_col = "features"
assembler = VectorAssembler(inputCols=feature_columns, outputCol=vector_col)
df_vector = assembler.transform(df_normalized).select(vector_col)

# Creating the correlation matrix
correlation_matrix = Correlation.corr(df_vector, vector_col).head()[0]

# Convert correlation matrix to dense matrix (for easier viewing)
correlation_matrix_dense = correlation_matrix.toArray()

# Threshold for removing highly correlated columns
threshold = 0.9
highly_correlated_pairs = []

for i in range(len(correlation_matrix_dense)):
    for j in range(i + 1, len(correlation_matrix_dense)):
        if abs(correlation_matrix_dense[i, j]) > threshold:
            highly_correlated_pairs.append((feature_columns[i], feature_columns[j]))

print("Highly correlated feature pairs:", highly_correlated_pairs)

# Removing one feature from each pair (this example drops the second feature in the pair)
features_to_drop = {pair[1] for pair in highly_correlated_pairs}  # Drop second feature of each pair
remaining_features = [f for f in feature_columns if f not in features_to_drop]

# Creating a new DataFrame with only non-correlated features
df_reduced = df_normalized.select(remaining_features)

# Show the final DataFrame without highly correlated features
print("Final DF without highly correlated columns")
df_reduced.show()


# BoxCox Transformation (Alternative to Standard Scaling) for correcting skewness

df_boxcox = df_encoded_spark.toPandas()

def apply_boxcox(df):
    transformed_df = df
    lambdas = {}  # Lambda values for each column

    for col_name in df.columns:
        if df[col_name].var() < 1:
            transformed_df[col_name] = df[col_name]
            lambdas[col_name] = None
        else:
        # Find the minimum value in the column
            min_value = df[col_name].min()

        # If there are non-positive values, shift the data so all values are positive
            shift_value = 1 - min_value if min_value <= 0 else 0

        # Applying Box-Cox transformation after shifting the column values
            transformed_df[col_name], lmbda = boxcox(df[col_name] + shift_value)
            lambdas[col_name] = lmbda

    return transformed_df, lambdas

# Apply Box-Cox to all numeric columns in the Pandas DataFrame
df_transformed, lambdas = apply_boxcox(df_boxcox)

# Convert the transformed Pandas DataFrame back to Spark DataFrame
df_transformed_spark = spark.createDataFrame(df_transformed)

# Show the transformed DataFrame
print("BoxCox transformed data")
df_transformed_spark.show()

## End of extra preprocessing steps----------------------------------------------------

# ML algorithms applied

df_normalized = scaled_data.select(
    F.col("tavg"),  # Retain the original `tavg` column
    *[F.col("features_array")[i].alias(name) for i, name in enumerate(normalized_columns)]
)

feature_columns = df_normalized.columns


def ml_pipeline(model, model_name, **kwargs):
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
    model_init = model(featuresCol = "features", labelCol = "tavg", **kwargs)
    
    print(f"{model} Initialized!")
    
    train, test = df_normalized.randomSplit([0.8, 0.2], seed = 123)
    pipeline = Pipeline(stages = [assembler, model_init])
    model_fit = pipeline.fit(train)
    predictions = model_fit.transform(test)

    rmse_eval = RegressionEvaluator(labelCol = "tavg", predictionCol = "prediction", metricName = "rmse")
    r2_eval = RegressionEvaluator(labelCol = "tavg", predictionCol = "prediction", metricName = "r2")

    rmse = rmse_eval.evaluate(predictions)
    r2 = r2_eval.evaluate(predictions)

    print(f"RMSE for {model_name}: {rmse}")
    print(f"R2 for {model_name}: {r2}\n")

    return rmse, r2

# Gradient Boosted Tree Regression
gbt_rmse, gbt_r2 = ml_pipeline(GBTRegressor, "Gradient Boosted Tree Regression", maxIter = 10)

# Decision Tree Regression
dt_rmse, dt_r2 = ml_pipeline(DecisionTreeRegressor, "Decision Tree Regression")

# Elastic Net (Linear Regression)
enet_rmse, enet_r2 = ml_pipeline(LinearRegression, "Elastic Net Linear Regression", maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)

# Random Forest Regression
rf_rmse, rf_r2 = ml_pipeline(RandomForestRegressor, "Random Forest Regression")



print(f"RMSE for Gradient Boosted Tree Regression: {gbt_rmse}")
print(f"R2 for Gradient Boosted Tree Regression: {gbt_r2}\n")


print(f"RMSE for Decision Tree Regression: {dt_rmse}")
print(f"R2 for Decision Tree Regression: {dt_r2}\n")


print(f"RMSE for Elastic Net Linear Regression: {enet_rmse}")
print(f"R2 for Elastic Net Linear Regression: {enet_r2}\n")

print(f"RMSE for Random Forest Regression: {rf_rmse}")
print(f"R2 for Random Forest Regression: {rf_r2}\n")




# Stopping the spark session
spark.stop()
