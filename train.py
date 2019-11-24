##################
##################
import mlflow
import pyspark.sql 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
import sys
from pyspark.sql import SparkSession

def main():

  spark = SparkSession\
      .builder\
      .appName("MLflow_local_example")\
      .getOrCreate()

  df = spark.read.parquet("/bmathew/mlflow_projects/atm_fraud/atm_transactions.snappy.parquet")
  df.createOrReplaceTempView("my_table")

  df2 = spark.sql("SELECT CAST(visit_id as INT),CAST(customer_id as INT),CAST(atm_id as INT),withdrawl_or_deposit,CAST(amount as decimal),year,month,day,hour,min,sec,fraud_report,first_name,last_name,city_state_zip,pos_capability,checking_savings,offsite_or_onsite,bank,credit_card_hash,last_4_card_digits FROM my_table")

  (trainingData, testData) = df2.randomSplit([0.70, 0.30], seed = 32)

  categoricals = ["withdrawl_or_deposit", "checking_savings", "pos_capability", "offsite_or_onsite", "bank"]
  indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx"), categoricals))
  featureCols = list(map(lambda c: c+"_idx", categoricals)) + ["amount", "day", "month", "year", "hour", "min"]

  maxDepth = int(sys.argv[1])
  numTrees = int(sys.argv[2])

  with mlflow.start_run(experiment_id=4879538):
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=maxDepth, numTrees=numTrees)
    stages = indexers + [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="fraud_report", outputCol="label")]
    pipeline = Pipeline(stages=stages+[rf])
    rfModel = pipeline.fit(trainingData)
    predictions = rfModel.transform(testData)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    evaluator.evaluate(predictions)
    auc = evaluator.evaluate(predictions)
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("numTrees", numTrees)
    mlflow.log_param("features", featureCols)
    mlflow.log_metric("auc", auc)
    mlflow.log_artifact("/dbfs/bmathew/mlflow_projects/atm_fraud/atm_transactions.snappy.parquet")

    from mlflow import spark
    mlflow.spark.log_model(rfModel, "RFMachineFailureModel_log")

if __name__ == "__main__":
  main()
