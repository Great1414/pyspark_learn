#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@license: (C) Copyright 2019-2020, 深圳市通用互联科技有限责任公司. 
@contact: renyw@gidomino.com
@file: test.py
@time: 2019-05-10 15:59
@desc: 在此写上代码文件的功能描述
'''
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext()
SparkSession(sc)
bdf = sc.parallelize([
Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()
blor = LogisticRegression(regParam=0.01, weightCol="weight")
blorModel = blor.fit(bdf)
test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, 1.0))]).toDF()
result = blorModel.transform(test0)
result.show()
