package Classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{ClassificationModel, NaiveBayes, SVMWithSGD, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater, SimpleUpdater, Updater}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Gini, Impurity, Entropy}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by yuch on 2016/5/8.
  * spark机器学习，分类
  */
object FeatureExtraction {

  //四个模型
  def ALG(): Unit ={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("E:\\downloads\\google-Chrome-download\\train.tsv")
    val records = rawData.map(_.trim.split("\t"))
    //    println(records.first().foreach(t => print(t+"  ")))

    //LabeledPoint 数据输入形式 普通分类模型
    val data =records.map{t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()


    //朴素贝叶斯 分类模型（因为不能有负数）
    val nbdata =records.map{ t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble).map(d =>
        if (d<0)0.0 else d)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()

    val numIterations = 10
    val maxTreeDepth = 5

    //逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(data,numIterations)

    //svm模型
    val svmModel = SVMWithSGD.train(data,numIterations)

    //朴素贝叶斯模型
    val nbModel = NaiveBayes.train(nbdata)

    //决策树模型
    val dtModel = DecisionTree.train(data,Algo.Classification,Entropy,maxTreeDepth)

    //    val dataPoint = data.take(100)
    //    val prediction = lrModel.predict(dataPoint.features)
    //    val prediction1 = svmModel.predict(dataPoint.features)
    //    val prediction2 = nbModel.predict(dataPoint.features)
    //    val prediction3 = dtModel.predict(dataPoint.features)
    //    print("真实的"+dataPoint.label+"  lrModel:"+prediction+"  svmModel:"+prediction1+"  nbModel:"+prediction2+"  dtModel:"+prediction3)

    //    val prediction4 = dataPoint.map{t=>
    //      nbModel.predict(t.features)
    //    }.foreach(t => print(t+" "))


    //逻辑回归的正确数目
    /*val lrTotalCorrect = data.map{ point =>
      if(lrModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    //SVM的正确数目
    val svmTotalCorrect = data.map{ point =>
      if(svmModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    //NB的正确数目
    val nbTotalCorrect = data.map{ point =>
      if(nbModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    //NB的正确数目
    val dtTotalCorrect = data.map{ point =>
      if(dtModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    print("LR:"+lrTotalCorrect/data.count()+"\n"+
          "SVM:"+svmTotalCorrect/data.count()+"\n"+
          "nb:"+nbTotalCorrect/data.count()+"\n"+
          "dt:"+dtTotalCorrect/data.count()+"\n"
    )*/

    //查找metirc
    val metric = Seq(lrModel,svmModel,nbModel).map{ model =>
      val scoreAndLabels = data.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getName,"PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }

    val dtmetric = Seq(dtModel).map{ model =>
      val scoreAndLabels = data.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getName,"PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }
    print(metric,dtmetric)
  }

  //数据标准化
  def featuresNormalization={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("E:\\downloads\\google-Chrome-download\\train.tsv")
    val records = rawData.map(_.trim.split("\t"))
    //    println(records.first().foreach(t => print(t+"  ")))

    //LabeledPoint 数据输入形式 普通分类模型
    val data =records.map{t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()

    val vectors = data.map{lp => lp.features}
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
//    print("count:"+matrixSummary.count+"\n"+
//          "max:"+matrixSummary.max+"\n"+
//          "mean:"+matrixSummary.mean+"\n"+
//          "min:"+matrixSummary.min+"\n"+
//          "normL1:"+matrixSummary.normL1+"\n"+
//          "normL2:"+matrixSummary.normL2+"\n"+
//          "numNonzeros:"+matrixSummary.numNonzeros+"\n"+
//          "variance:"+matrixSummary.variance)

    val scaler = new StandardScaler(true,true).fit(vectors)
    val scaledData = data.map{lp => LabeledPoint(lp.label,scaler.transform(lp.features))}
    scaledData.cache()
//    println(data.first().features)
//    println(scaledData.first().features)

    //逻辑回归模型（非标准化数据）
    val lrModel = LogisticRegressionWithSGD.train(data,10)

    //逻辑回归模型（标准化数据）
    val lrModelNormalizer = LogisticRegressionWithSGD.train(scaledData,10)

    val dtmetric = Seq(lrModel).map{ model =>
      val scoreAndLabels = data.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getName,"PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }
    val dtmetricNormalizer = Seq(lrModelNormalizer).map{ model =>
      val scoreAndLabels = data.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getName,"PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }

    print(dtmetric,dtmetricNormalizer)
  }

  //添加特征
  def addFeature={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("E:\\downloads\\google-Chrome-download\\train.tsv")
    val records = rawData.map(_.trim.split("\t"))
    //LabeledPoint 数据输入形式 普通分类模型
    val data =records.map{t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()

    val categories = records.map(r => r(3)).distinct.collect().zipWithIndex.toMap
    print(categories)
    val num = categories.size
    val dataCategories = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"",""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](num)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4,r.size-1).map(d =>
      if(d == "?") 0.0 else d.toDouble)
      val features = otherFeatures ++ categoryFeatures
      LabeledPoint(label,Vectors.dense(features))
    }
    print(dataCategories.first())

    //标准化
    val scaler = new StandardScaler(true,true).fit(data.map(a => a.features))
    val scaledData = data.map{lp => LabeledPoint(lp.label,scaler.transform(lp.features))}

    //增加维度标准化
    val scalerAdd = new StandardScaler(true,true).fit(dataCategories.map(a => a.features))
    val scaledDataAdd = dataCategories.map{lp => LabeledPoint(lp.label,scalerAdd.transform(lp.features))}

    //逻辑回归模型（标准化数据）
    val lrModelNormalizer = LogisticRegressionWithSGD.train(scaledData,10)

    //逻辑回归模型（增加维度标准化数据）
    val lrModelNormalizerAdd = LogisticRegressionWithSGD.train(scaledDataAdd,10)

    //标准化
    val dtmetric = Seq(lrModelNormalizer).map{ model =>
      val scoreAndLabels = scaledData.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      ("dtmetric","PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }

    //增加维度标准化
    val dtmetricAdd = Seq(lrModelNormalizerAdd).map{ model =>
      val scoreAndLabels = scaledDataAdd.map{ t =>
        (model.predict(t.features),t.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      ("dtmetricAdd","PR=="+metrics.areaUnderPR(),"ROC=="+metrics.areaUnderROC())
    }

    print(dtmetric,dtmetricAdd)

  }

  //参数优化
  def paramOptimize={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("E:\\downloads\\google-Chrome-download\\train.tsv")
    val records = rawData.map(_.trim.split("\t"))
    //LabeledPoint 数据输入形式 普通分类模型
    val data =records.map{t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()

    //标准化
    val scaler = new StandardScaler(true,true).fit(data.map(a => a.features))
    val scaledData = data.map{lp => LabeledPoint(lp.label,scaler.transform(lp.features))}

    //增加循环次数
//    val iterResults = Seq(1,5,10,50).map{param =>
//      val model = trainWithParams(scaledData,0.0,param,new SimpleUpdater,1.0)
//      createMetircs(param.toString,scaledData,model)
//    }

    //增加步长
    val iterResults1 = Seq(0.001,0.01,0.1,1.0,10.0).map{param =>
      val model = trainWithParams(scaledData,0.0,10,new SimpleUpdater,param)
      createMetircs(param.toString,scaledData,model)
    }

    //正则化
    val iterResults2 = Seq(0.001,0.01,0.1,1.0,10.0).map{param =>
      val model = trainWithParams(scaledData,0.0,10,new SquaredL2Updater,param)
      createMetircs(param.toString,scaledData,model)
    }

    //正则化
    val iterResults3 = Seq(0.001,0.01,0.1,1.0,10.0).map{param =>
      val model = trainWithParams(scaledData,0.0,10,new L1Updater,param)
      createMetircs(param.toString,scaledData,model)
    }

    Seq(iterResults1,iterResults2,iterResults3).foreach{a => a.foreach{case (label,pr,auc) =>
    println(f"$label,pr=${pr*100}%2.2f%%,auc=${auc * 100}%2.2f%%")
    }}

  }

  //决策树树深度和信息增益调优
  def treeDepthAndImpurity={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("E:\\downloads\\google-Chrome-download\\train.tsv")
    val records = rawData.map(_.trim.split("\t"))
    //LabeledPoint 数据输入形式 普通分类模型
    val data =records.map{t =>
      val trimmed = t.map(_.replaceAll("\"",""))
      val label = trimmed(t.size-1).toInt
      val features = trimmed.slice(4,t.size-1).map(d =>
        if(d=="?") 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }.cache()

    val dtResultsEntropy = Seq(1,2,3,4,5,10,20).map{ param =>
      val model = trainDTWithParams(data,param,Entropy)
      val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (score,point.label)
      }
      val metric = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree deep",metric.areaUnderPR(),metric.areaUnderROC())
    }

    val dtResultsEntropy1 = Seq(1,2,3,4,5,10,20).map{ param =>
      val model = trainDTWithParams(data,param,Gini)
      val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (score,point.label)
      }
      val metric = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree deep",metric.areaUnderPR(),metric.areaUnderROC())
    }

    dtResultsEntropy.foreach{case (label,pr,auc) =>
      println(f"$label,pr=${pr*100}%2.2f%%,auc=${auc * 100}%2.2f%%")
    }

    print("================================")
    dtResultsEntropy1.foreach{case (label,pr,auc) =>
      println(f"$label,pr=${pr*100}%2.2f%%,auc=${auc * 100}%2.2f%%")
    }

  }

  //参数训练模型
  def trainWithParams(input: RDD[LabeledPoint],regParam:Double,numIterations:Int,update:Updater,stepSize:Double) ={
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer.setNumIterations(numIterations)
      .setUpdater(update)
      .setUpdater(update)
      .setRegParam(regParam)
      .setStepSize(stepSize)
    lr.run(input)
  }

  //创造度量函数
  def createMetircs(label:String,data:RDD[LabeledPoint],model:ClassificationModel) ={
    val scoreAndLabels = data.map{point =>
      (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label,metrics.areaUnderPR(),metrics.areaUnderROC())
  }

  def trainDTWithParams(input:RDD[LabeledPoint],maxDepth:Int,impurity:Impurity)={
    DecisionTree.train(input,Algo.Classification,impurity,maxDepth)
  }

  def main(args: Array[String]): Unit = {
    treeDepthAndImpurity
  }
}
