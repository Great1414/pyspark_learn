#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@license: (C) Copyright 2019-2020, GI.
@contact: renyw@gidomino.com
@file: lr_classifer.py
@time: 2019-05-10 10:24
@desc: logistic for classifier by pyspark
'''
import os
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#0-config
os.environ["HADOOP_HOME"] = "F:\\hadoop-common-2.2.0-bin"
os.environ["SPARK_HOME"] = r"F:\pycharm_project_location\spark-2.4.1-bin-hadoop2.7"
#1-connect
spark = SparkSession\
    .builder\
    .appName("lr_classifer app")\
    .master("local[2]")\
    .getOrCreate()
#2-data process
#2.1-read
dataF = spark.read.csv("../data/cluster_data.csv", header=True, inferSchema=True)
#2.2-duplicate/miss/outlier/similarity

#2.3-label,feature format for ml/encoder/scaler/
def data_to_vector(data):
    return data.rdd.map(lambda x: [x["label"], Vectors.dense(x["displacement"])]).toDF(["label", "features"])

# def test_to_vector(data):
#     return data.rdd.map(lambda x: [Vectors.dense(x["displacement"])]).toDF(["features"])
trainData, testData = dataF.select("displacement", "label").randomSplit([0.75, 0.25], seed=10)

trainDataVector = data_to_vector(trainData)
testDataVector = data_to_vector(testData)

# testDataVector.show()
# labeledIndex = StringIndexer(inputCol="label", outputCol="labeled")
# featuredIndex = VectorIndexer(inputCol="displacement", outputCol="featuredDis")
#sampleby for imbalance
#model
logistiClass = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction",
                                  )
pipeline = Pipeline(stages=[logistiClass])
# logistiClassModel = logistiClass.fit(trainDataVector)
# result = logistiClassModel.transform(testDataVector)
# model = pipeline.fit(trainDataVector)
# predictions = model.transform(testDataVector)
#evaluate/crossvalid

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
# accuary = evaluator.evaluate(predictions)
paramGrid = ParamGridBuilder()\
             .addGrid(logistiClass.regParam, [0.01, 0.5, 2.0])\
             .addGrid(logistiClass.elasticNetParam, [0.0, 0.5, 1.0])\
             .addGrid(logistiClass.maxIter, [1, 5, 10]).build()

crossVal = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
logisticModel = crossVal.fit(trainDataVector)
#best model
bestParams = [
    (
        [
            {key.name:paramValue}
            for key, paramValue in zip(params.keys(), params.values())
        ], metric
    )
    for params, metric in zip(logisticModel.getEstimatorParamMaps(),
                              logisticModel.avgMetrics)
]
bestParams = sorted(bestParams, key=lambda x: x[1], reverse=True)[0]
print(bestParams)
treeModel = logisticModel.bestModel
print(logisticModel.avgMetrics)
print(treeModel)

predictions = logisticModel.transform(testDataVector)
accuary = evaluator.evaluate(predictions)
print("accuary:", accuary)
#save
path = "./logisticModel"
treeModel.write().overwrite().save(path)
model = PipelineModel.load(path)
result = model.transform(testDataVector)
print(result)







