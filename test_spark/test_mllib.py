#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@file: test_mllib.py
@time: 2019-05-06 17:33
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
os.environ["HADOOP_HOME"] = "F:\\hadoop-common-2.2.0-bin"
os.environ["SPARK_HOME"] = r"F:\pycharm_project_location\spark-2.4.1-bin-hadoop2.7"
sc = SparkContext("local[2]", "test_app")
spark = SparkSession(sc)
# spark = SparkSession.builder.appName("test_app").master("local[2]").getOrCreate()
train_data = spark.read.csv("../data/churn-bigml-80.csv", header=True, inferSchema=True)
test_data = spark.read.csv("../data/churn-bigml-20.csv", header=True,inferSchema="true")
# train_data.cache()
# test_data.cache()
# train_data.printSchema()
# 缺失值分析
# print(train_data.describe().toPandas())
# print(train_data.toPandas().count()) #no null
# 相关性分析,去除无用字段
# numeric_features = [each[0] for each in train_data.dtypes if each[1] == "int" or each[1] == "double"]
# sample_data = train_data.select(numeric_features).sample(False, 0.1).toPandas()
# pd.scatter_matrix(sample_data, figsize=(12, 12))
# plt.show()
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
"""
# MLlib
def labelData(data):
    return data.map(lambda x: LabeledPoint(x[-1], x[:-1]))

train_data, valid_data = labelData(train_data.rdd).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(train_data, numClasses= 2, maxDepth=2,
                                     categoricalFeaturesInfo= {1:2, 2:2},
                                     impurity="gini", maxBins= 32)
# print(model.toDebugString())
# MLlib evaluate
def getPredictionLabels(model, valid_data):
    predictions = model.predict(valid_data.map(lambda r:r.features))
    return predictions.zip(valid_data.map(lambda r:r.label))

def printMetrics(result):
    metrics = MulticlassMetrics(result)
    print("\nPrecision of True\n", metrics.precision(1))
    print("\nPrecision of False\n", metrics.precision(0))
    print("\nRecall of True\n", metrics.recall(1))
    print("\nRecall of False\n", metrics.recall(0))
    print("\nF1 score\n", metrics.fMeasure())
    print("\nConfusion Matrix\n", metrics.confusionMatrix().toArray())

result = getPredictionLabels(model, valid_data)
printMetrics(result)
"""
# 样本分布,False与True接近1:6，样本不平静
# train_data.groupBy("Churn").count().show()
# 分层采样
def labelData(data):
    return data.map(lambda x: LabeledPoint(x[-1], x[:-1]))
stratified_train = train_data.sampleBy("Churn", fractions={'0.0': 0.18, '1.0': 1.0})
train_data.groupBy("Churn").count().show()
stratified_train.groupBy("Churn").count().show()
training_data, valid_data = labelData(stratified_train.rdd).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,
                                     categoricalFeaturesInfo={1:2, 2:2},
                                     impurity='gini', maxBins=32)
def getPredictionLabels(model, valid_data):
    predictions = model.predict(valid_data.map(lambda r:r.features))
    return predictions.zip(valid_data.map(lambda r:r.label))

def printMetrics(result):
    metrics = MulticlassMetrics(result)
    print("\nPrecision of True\n", metrics.precision(1))
    print("\nPrecision of False\n", metrics.precision(0))
    print("\nRecall of True\n", metrics.recall(1))
    print("\nRecall of False\n", metrics.recall(0))
    print("\nF1 score\n", metrics.fMeasure())
    print("\nConfusion Matrix\n", metrics.confusionMatrix().toArray())

result = getPredictionLabels(model, valid_data)
printMetrics(result)
predictions_and_labels = getPredictionLabels(model, valid_data)
printMetrics(predictions_and_labels)




