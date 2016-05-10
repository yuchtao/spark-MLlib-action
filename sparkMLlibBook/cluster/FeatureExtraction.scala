package cluster

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by yuch on 2016/5/8.
  * spark机器学习，分类
  */
object FeatureExtraction {

  def Kmeans(): Unit ={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    //1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    val movies = sc.textFile("F:\\迅雷下载\\ml-100k\\u.item")

    //unknown|0   Western|18    War|17
    val genres = sc.textFile("F:\\迅雷下载\\ml-100k\\u.genre")

    val genresMap = genres.filter(!_.isEmpty).map(line =>
      line.split("\\|")).map(t =>(t(1),t(0))).collectAsMap()

    val titlsAndGenres = movies.map(_.split("\\|")).map{array =>
      val genres = array.slice(5,array.size)
      val genresAssigned = genres.zipWithIndex.filter{case (g,idx) =>
        g == "1"}.map{case (g,idx) => genresMap(idx.toString)}
      (array(0).toInt,(array(1),genresAssigned))
    }

    //打印出来结果
    //    val a = titlsAndGenres.map{case (a,(b,c)) => c}
    //    a.first().foreach(println)

    //训练模型
    val rawData = sc.textFile("F:\\迅雷下载\\ml-100k\\u.data")
    val rawRating = rawData.map(_.trim.split("\t").take(3))

    val ratings = rawRating.map{case Array(usr,movie,rating) =>
      Rating(usr.toInt,movie.toInt,rating.toDouble)}

    //得到模型
    val alsModel = ALS.train(ratings,50,10,0.1)

    val movieFactors = alsModel.productFeatures.map{ case(id,factor) =>
      (id,Vectors.dense(factor))
    }

    //电影参数
    val movieVectors = movieFactors.map(_._2)

    val userFactors = alsModel.userFeatures.map{ case(id,factor) =>
      (id,Vectors.dense(factor))
    }

    //用户参数
    val userVectors = userFactors.map(_._2)

    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()

    //均值
    //    println(movieMatrixSummary.mean)
    //方差
    //    println(movieMatrixSummary.variance)

    val numClusters = 5
    val numIterations = 10
    val numRuns = 3

    val movieClusterModel = KMeans.train(movieVectors,numClusters,numIterations,numRuns)

    val useClusterModel = KMeans.train(userVectors,numClusters,numIterations,numRuns)

    //预测
    //    val movie = movieVectors.first()
    //    val movieCluster  = movieClusterModel.predict(movieVectors)
    //    println(movieCluster.take(10).mkString("  "))
    val movieCost = movieClusterModel.computeCost(movieVectors)
    val userCost = useClusterModel.computeCost(userVectors)
    //    println(movieCost , userCost)

    /*val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6,0.4),123)
    val (trainMovies,testMovies) = (trainTestSplitMovies(0),trainTestSplitMovies(1))

    val costMovies = Seq(2,3,4,5,10,20,50).map{k =>
      (k,KMeans.train(trainMovies,k,numIterations,numRuns).computeCost(testMovies))
    }
    costMovies.foreach{case (k,cost) =>
      println(k+"    "+cost)
    }*/

    val trainTestSplitMovies = userVectors.randomSplit(Array(0.6,0.4),123)
    val (trainMovies,testMovies) = (trainTestSplitMovies(0),trainTestSplitMovies(1))

    val costMovies = Seq(2,3,4,5,10,20,50).map{k =>
      (k,KMeans.train(trainMovies,k,numIterations,numRuns).computeCost(testMovies))
    }
    costMovies.foreach{case (k,cost) =>
      println(k+"    "+cost)
    }

  }

  def main(args: Array[String]): Unit = {
    Kmeans
  }
}

