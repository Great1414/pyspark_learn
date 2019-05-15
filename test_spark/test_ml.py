#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@file: test_ml.py
@time: 2019-05-08 18:22
@desc: 在此写上代码文件的功能描述
'''
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.types as typ
import matplotlib.pylab as plt
import os
import pandas as pd
import pyspark.sql.functions as fn
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
os.environ["HADOOP_HOME"] = "F:\\hadoop-common-2.2.0-bin"
os.environ["SPARK_HOME"] = r"F:\pycharm_project_location\spark-2.4.1-bin-hadoop2.7"
sc = SparkContext("local[2]", "test_app")
spark = SparkSession(sc)
# spark = SparkSession.builder.appName("test_app").master("local[2]").getOrCreate()
train_data = spark.read.csv("../data/churn-bigml-80.csv", header=True, inferSchema=True)
test_data = spark.read.csv("../data/churn-bigml-20.csv", header=True,inferSchema="true")
# train_data.cache()
# test_data.cache()
# 类别转换为数值，0-1转码
map_transpose = {"Yes": 1.0, True: 1.0, "No": 0.0, False: 0.0}
toNum = UserDefinedFunction(lambda x: map_transpose[x])
train_data = train_data.drop("State").drop("Area code").drop("Total day charge")\
    .drop("Total eve charge").drop("Total night charge").drop("Total intl charge")\
    .withColumn("Churn", toNum(train_data["Churn"]))\
    .withColumn("International plan", toNum(train_data["International plan"]))\
    .withColumn('Voice mail plan', toNum(train_data['Voice mail plan'])).cache()

test_data = test_data.drop("State").drop("Area code").drop("Total day charge")\
    .drop("Total eve charge").drop("Total night charge").drop("Total intl charge")\
    .withColumn("Churn", toNum(test_data["Churn"]))\
    .withColumn("International plan", toNum(test_data["International plan"]))\
    .withColumn('Voice mail plan', toNum(test_data['Voice mail plan'])).cache()

def vectorizeData(data):
    return data.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).toDF(["label", "features"])

vectorized_train = vectorizeData(train_data.rdd)
vectorized_train.show()
labelIndexer = StringIndexer(
    inputCol="label", outputCol="indexedLabel").fit(vectorized_train)
featureIndexer = VectorIndexer(
    inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(vectorized_train)

dTree = DecisionTreeClassifier(featuresCol="indexedFeatures", labelCol="indexedLabel")
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dTree])
paraGrid = ParamGridBuilder().addGrid(dTree.maxDepth, [2,4,6,8]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",metricName="f1")
crossVal = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paraGrid,
    evaluator=evaluator,
    numFolds=4
)

cv_model = crossVal.fit(train_data)
best_tree = cv_model.bestModel.stages[2]
print(best_tree)
