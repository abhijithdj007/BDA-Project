# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Food_Inspections.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

# Function to map Pass/Fail to 1.0 or 0.0 (and -1 for other values)
def labelForResults(s):
    if s == 'Fail':
        return 0.0
    elif s == 'Pass w/ Conditions' or s == 'Pass':
        return 1.0
    else:
        return -1.0
      

udfsomefunc = F.udf(labelForResults, DoubleType())
# Create a view or table

labelled_df = df.withColumn("labelled", udfsomefunc("Results"))

temp_table_name = "Food_Inspections_csv"

labelled_df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `Food_Inspections_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "Food_Inspections_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT results, COUNT(*) as totals FROM Food_Inspections_csv GROUP BY results ORDER BY totals DESC

# COMMAND ----------

overallByType = spark.sql("SELECT Inspection_Type, COUNT(*) AS counts FROM Food_Inspections_csv GROUP BY Inspection_Type ORDER BY counts DESC, Inspection_Type")

display(overallByType)

# COMMAND ----------

topFailuresByType = spark.sql("SELECT Inspection_Type, COUNT(*) AS counts FROM Food_Inspections_csv WHERE results = 'Fail' GROUP BY Inspection_Type ORDER BY counts DESC, Inspection_Type")

display(topFailuresByType)

# COMMAND ----------

overallByFacility = spark.sql("SELECT Facility_Type, COUNT(*) AS counts FROM Food_Inspections_csv GROUP BY Facility_Type ORDER BY counts DESC, Facility_Type")

display(overallByFacility)

# COMMAND ----------

topFailuresByFacility = spark.sql("SELECT Facility_Type, COUNT(*) AS counts FROM Food_Inspections_csv WHERE results = 'Fail' GROUP BY Facility_Type ORDER BY counts DESC, Facility_Type")

display(topFailuresByFacility)

# COMMAND ----------

temp_labeledData = spark.sql("SELECT Facility_Type, Inspection_Type, Zip, labelled as label FROM Food_Inspections_csv WHERE labelled >= 0 and Facility_Type != 'NULL' and Inspection_Type != 'NULL' limit 15000")

labeledData = temp_labeledData.filter("Facility_Type != ' '").filter("Inspection_Type != ' '")
labeledData.show()


# COMMAND ----------

# Use indexers to convert from string values to a numeric index value
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

facilityIndexer = StringIndexer(inputCol="Facility_Type", outputCol="FacilityTypeIndex")
inspectionIndexer = StringIndexer(inputCol="Inspection_Type", outputCol="InspectionTypeIndex")
zipIndexer = StringIndexer(inputCol="Zip", outputCol="ZipIndex")

# Run the indexers to create a new dataframe
pipeline = Pipeline(stages=[facilityIndexer, inspectionIndexer, zipIndexer])
indexedData = pipeline.fit(labeledData).transform(labeledData)

indexedData.count()

# COMMAND ----------

# Convert from several discrete feature columns to a single vector feature column
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["FacilityTypeIndex", 
                                       "InspectionTypeIndex", 
                                       "ZipIndex"], 
                            outputCol="features")
preparedData = assembler.transform(indexedData)

preparedData.show()

# COMMAND ----------

# Split the sample data into 80% training set, 20% scoring/evaluation set

(trainingData, scoringData) = preparedData.randomSplit([0.8, 0.2], seed = 100)

print("Training: " + str(trainingData.count()) + ' records.')
print("Scoring: " + str(scoringData.count()) + ' records.')

# COMMAND ----------

import mlflow
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')

# add empty parameter grid 
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [4, 8]) # max depth parameter
             .addGrid(dt.maxBins, [200,220,250]) # max bins Parameter 
             .build())

# create evaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

# create cross validation object
crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=10) 

# run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

# make predictions on test data 
prediction_mcdt = cvModel.transform(scoringData)

prediction_mcdt.createOrReplaceTempView('dtpredictions')

# Display the success rate
numSuccesses = prediction_mcdt.where("""(prediction = 0 AND label = 0) OR (prediction = 1 AND label = 1)""").count()
numInspections = prediction_mcdt.count()

print("There were", numInspections, "inspections and there were", numSuccesses, "successful DecisionTreeClassifier predictions")
print("This is a", str((float(numSuccesses) / float(numInspections)) * 100) + "%", "success rate")

# COMMAND ----------

# Train the ML model
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
nbModel = nb.fit(trainingData)

# Score/evaluate the model
predictions = nbModel.transform(scoringData)
# predictions.printSchema()
predictions.registerTempTable('predictions')

# Display the success rate
numSuccesses = predictions.where("""(prediction = 0 AND label = 0) OR (prediction = 1 AND label = 1)""").count()
numInspections = predictions.count()

print("There were", numInspections, "inspections and there were", numSuccesses, "successful NaiveBayes predictions")
print("This is a", str((float(numSuccesses) / float(numInspections)) * 100) + "%", "success rate")

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier

mcpc = MultilayerPerceptronClassifier(featuresCol='features', labelCol='label',maxIter= 100, layers=[3, 4, 3, 3])

# add parameter grid 
paramGrid = ParamGridBuilder().build()

# create evaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

crossval = CrossValidator(estimator=mcpc,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=10) 

# run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

# make predictions on test data
prediction_rf = cvModel.transform(scoringData)

prediction_rf.registerTempTable('predictions_rf')

# Display the success rate
numSuccesses = prediction_rf.where("""(prediction = 0 AND label = 0) OR (prediction = 1 AND label = 1)""").count()
numInspections = prediction_rf.count()

print("There were", numInspections, "inspections and there were", numSuccesses, "successful MultilayerPerceptronClassifier predictions")
print("This is a", str((float(numSuccesses) / float(numInspections)) * 100) + "%", "success rate")

# COMMAND ----------

# %sql
# SELECT count(*) AS cnt FROM Predictions WHERE prediction = 1 AND label = 1

true_positive = spark.sql("SELECT count(*) AS cnt FROM predictions WHERE prediction = 1 AND label = 1")

# COMMAND ----------

# %sql
# SELECT count(*) AS cnt FROM Predictions WHERE prediction = 1 AND label = 0

false_positive = spark.sql("SELECT count(*) AS cnt FROM predictions WHERE prediction = 1 AND label = 0")

# COMMAND ----------

# %sql
# SELECT count(*) AS cnt FROM Predictions WHERE prediction = 0 AND label = 0

true_negative = spark.sql("SELECT count(*) AS cnt FROM predictions WHERE prediction = 0 AND label = 0")

# COMMAND ----------

# %sql
# SELECT count(*) AS cnt FROM Predictions WHERE prediction = 0 AND label = 1

false_negative = spark.sql("SELECT count(*) AS cnt FROM predictions WHERE prediction = 0 AND label = 1")

# COMMAND ----------

# labels_score = {'True positive' : true_positive, 'False positive' : false_positive, 'True negative' : true_negative, 'False negative' : false_negative}
# display(labels_score)
df1 = spark.createDataFrame([('True positive', true_positive.first()['cnt']), ('False positive', false_positive.first()['cnt']), ('True negative', true_negative.first()['cnt']), ('False negative',false_negative.first()['cnt'])], ("labels", "size"))
display(df1)
