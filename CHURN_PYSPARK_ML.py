
import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init(r"C:\spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_hw") \
    .getOrCreate()

sct = spark.sparkContext

############################
# Exploratory Data Analysis
############################

spark_df = spark.read.csv("WEEK_11/datasets_11/churn2.csv", header=True, inferSchema=True)
spark_df
type(spark_df)

spark_df.dtypes

# Number of observation and variable
print("Shape: ", (spark_df.count(), len(spark_df.columns)))
# (10000, 14)

# Types of Variables
spark_df.printSchema()
spark_df.dtypes

# | -- RowNumber: integer(nullable=true)
# | -- CustomerId: integer(nullable=true)
# | -- Surname: string(nullable=true)
# | -- CreditScore: integer(nullable=true)
# |-- Geography: string (nullable = true)
# |-- Gender: string (nullable = true)
# |-- Age: integer (nullable = true)
# |-- Tenure: integer (nullable = true)
# |-- Balance: double (nullable = true)
# |-- NumOfProducts: integer (nullable = true)
# |-- HasCrCard: integer (nullable = true)
# |-- IsActiveMember: integer (nullable = true)
# |-- EstimatedSalary: double (nullable = true)
# |-- Exited: integer (nullable = true)

spark_df.show(5)

spark_df = spark_df.toDF(*[c.upper() for c in spark_df.columns])
spark_df.show(5)

# Summary statistics
spark_df.describe().show()

spark_df.describe(["AGE", "EXITED"]).show()

# Categorical variable class statistics
spark_df.groupby("EXITED").count().show()

# |EXITED|count|
# +------+-----+
# |     1| 2037|
# |     0| 7963|
# +------+-----+

# Unique classes
spark_df.select("EXITED").distinct().show()

# groupby transactions
spark_df.groupby("EXITED").count().show()
spark_df.groupby("EXITED").agg({"TENURE": "mean"}).show()

# +------+-----------------+
# |EXITED|      avg(TENURE)|
# +------+-----------------+
# |     1|4.932744231713304|
# |     0|5.033278914981791|

# Selecting of numeric variables
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()

spark_df.select(num_cols).describe().toPandas().transpose()

# Selecting of categoric variables
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Summary statistics of numeric variables with respect to EXITED
for col in [col.lower() for col in num_cols]:
    spark_df.groupby("EXITED").agg({col: "mean"}).show()

##################################################
# Data Preprocessing & Feature Engineering
##################################################

############################
# Missing Values
############################

from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T


############################
# Feature Interaction
############################

spark_df = spark_df.drop('ROWNUMBER', "CUSTOMERID", "SURNAME")

spark_df = spark_df.withColumn('CRSCORE_SALARY', spark_df.CREDITSCORE / spark_df.ESTIMATEDSALARY)
spark_df = spark_df.withColumn('CRSCORE_TENURE', spark_df.CREDITSCORE * spark_df.TENURE)
spark_df = spark_df.withColumn('TENURE_AGE', spark_df.TENURE / spark_df.AGE)
spark_df = spark_df.withColumn('CRSCORE_PRODUCTS', spark_df.CREDITSCORE / spark_df.NUMOFPRODUCTS)
spark_df = spark_df.withColumn('BALANCE_SALARY', spark_df.BALANCE / spark_df.ESTIMATEDSALARY)
spark_df.show(5)

############################
# Bucketization / Bining / Num to Cat
############################

# FOR AGE
spark_df.select('AGE').describe().toPandas().transpose()
spark_df.select("AGE").summary("count", "min", "25%", "50%","75%", "max").show()
bucketizer = Bucketizer(splits=[0, 35, 55, 75, 95], inputCol="AGE", outputCol="AGE_CAT")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('AGE_CAT', spark_df.AGE_CAT + 1)

spark_df.groupby("AGE_CAT").count().show()
spark_df.groupby("AGE_CAT").agg({'EXITED': "mean"}).show()

# FOR ESTIMATED SALARY
spark_df.select('ESTIMATEDSALARY').describe().toPandas().transpose()
spark_df.select("ESTIMATEDSALARY").summary("count", "min", "25%", "50%","75%", "max").show()
bucketizer = Bucketizer(splits=[0, 50000, 100000, 150000, 200000], inputCol="ESTIMATEDSALARY", outputCol="SALARY_CAT")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('SALARY_CAT', spark_df.SALARY_CAT + 1)

spark_df.groupby("SALARY_CAT").count().show()
spark_df.groupby("SALARY_CAT").agg({'EXITED': "mean"}).show()

# FOR TENURE
spark_df.select('TENURE').describe().toPandas().transpose()
spark_df.select("TENURE").summary("count", "min", "25%", "50%","75%", "max").show()
bucketizer = Bucketizer(splits=[0, 3, 5, 7, 10], inputCol="TENURE", outputCol="TENURE_CAT")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('TENURE_CAT', spark_df.TENURE_CAT + 1)

spark_df.groupby("TENURE_CAT").count().show()
spark_df.groupby("TENURE_CAT").agg({'EXITED': "mean"}).show()

spark_df.show(20)

# Float to integer
spark_df = spark_df.withColumn("AGE_CAT", spark_df["AGE_CAT"].cast("integer"))
spark_df = spark_df.withColumn("SALARY_CAT", spark_df["SALARY_CAT"].cast("integer"))
spark_df = spark_df.withColumn("TENURE_CAT", spark_df["TENURE_CAT"].cast("integer"))

spark_df.show(20)

############################
# Generating a variable with when (SEGMENT)
############################

spark_df = spark_df.withColumn('SEGMENT', when(spark_df['TENURE'] < 5, "SEG_B").otherwise("SEG_A"))

############################
# Label Encoding
############################

spark_df.show(5)

indexer = StringIndexer(inputCol="SEGMENT", outputCol="SEGMENT_LABEL")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("SEGMENT_LABEL", temp_sdf["SEGMENT_LABEL"].cast("integer"))
spark_df = spark_df.drop('SEGMENT')

indexer = StringIndexer(inputCol="GENDER", outputCol="GENDER_LABEL")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("GENDER_LABEL", temp_sdf["GENDER_LABEL"].cast("integer"))
spark_df = spark_df.drop('GENDER')

# OHE will be done for GEOGRAPHY, but it must pass through the label encoder first.

indexer = StringIndexer(inputCol="GEOGRAPHY", outputCol="GEOGRAPHY_LABEL")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("GEOGRAPHY_LABEL", temp_sdf["GEOGRAPHY_LABEL"].cast("integer"))
spark_df = spark_df.drop('GEOGRAPHY')

spark_df.show(5)

############################
# One Hot Encoding
############################
spark_df.show(5)

encoder = OneHotEncoder(inputCols=["AGE_CAT", "SALARY_CAT", "TENURE_CAT", "GEOGRAPHY_LABEL"], outputCols=["AGE_CAT_OHE", "SALARY_CAT_OHE", "TENURE_CAT_OHE", "GEOGRAPHY_LABEL_OHE"])
spark_df = encoder.fit(spark_df).transform(spark_df)

############################
# Defining TARGET
############################

stringIndexer = StringIndexer(inputCol='EXITED', outputCol='label')

temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
temp_sdf.show()
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)

############################
# Defining Features
############################

cols = ['CREDITSCORE', 'AGE', 'TENURE', 'BALANCE','NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'CRSCORE_SALARY', 'CRSCORE_TENURE', 'TENURE_AGE', 'CRSCORE_PRODUCTS', 'SEGMENT_LABEL', 'GENDER_LABEL', 'AGE_CAT_OHE', 'SALARY_CAT_OHE', 'TENURE_CAT_OHE', 'GEOGRAPHY_LABEL_OHE', 'BALANCE_SALARY']

# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="FEATURES")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("FEATURES", "label")
final_df.show(5)

# StandardScaler
scaler = StandardScaler(inputCol="FEATURES", outputCol="SCALED_FEATURES")
final_df = scaler.fit(final_df).transform(final_df)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

# Training Dataset Count: 6949
# Test Dataset Count: 3051

##################################################
# Modeling
##################################################

############################
# Logistic Regression
############################

log_model = LogisticRegression(featuresCol='FEATURES', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# 0.837430350704687

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))
# accuracy: 0.837430, precision: 0.854113, recall: 0.960132, f1: 0.815898, roc_auc: 0.657250

############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="FEATURES", labelCol="label")
gbm_model = gbm.fit(train_df) #modeli fit ediyoruz.
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# 0.8574237954768928

############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30, 40])
              .addGrid(gbm.maxIter, [10, 20, 30])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)

y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# 0.8613569321533924