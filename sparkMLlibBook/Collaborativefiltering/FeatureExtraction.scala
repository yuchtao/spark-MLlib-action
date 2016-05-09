package Collaborativefiltering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkContext, SparkConf}
import org.jblas.DoubleMatrix

/**
  * Created by yuch on 2016/5/8.
  * spark机器学习，协同过滤
  */
object FeatureExtraction {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("F:\\迅雷下载\\ml-100k\\u.data")
    val rawRating = rawData.map(_.trim.split("\t").take(3))

    //训练模型
    val rating = rawRating.map{case Array(usr,movie,rating) =>
      Rating(usr.toInt,movie.toInt,rating.toDouble)}
//    val modelImplicit = ALS.trainImplicit(rating,50,10,0.01,1)
    val model = ALS.train(rating,50,10,0.01)
//    print(modelImplicit.userFeatures.map(t => t._2.size).first())
//    val predictedRating = model.predict(249,241)
    val topKRecs = model.recommendProducts(789,10)

    //电影id到标题的映射
//    val movies = sc.textFile("F:\\迅雷下载\\ml-100k\\u.item")
//    val titles = movies.map(_.split("\\|").take(2)).map(t=>(t(0).toInt,t(1))).collectAsMap();
//    println(titles(1))

    //用户评价过多少电影
    val moviesForUser = rating.keyBy(_.user).lookup(789)
    val actualRating = moviesForUser.take(1)(0)
//    print(actualRating)

    //789用户的均方差
    val predictedRating = model.predict(789,actualRating.product)
//    print(predictedRating)
    val squaredError = math.pow(actualRating.rating-predictedRating,2.0)
//    print(squaredError)

    val userAndProducts = rating.map{case Rating(user,item,rating)
      => (user,item)}

    val predict = model.predict(userAndProducts).map{case Rating(user,item,rating) =>
      ((user,item),rating)}

    val actualAndPredict = rating.map{case Rating(user,item,rating)
    => ((user,item),rating)}.join(predict)


    //用户评级前十名的电影
//    val moviesForUserOrder = moviesForUser.sortBy(-_.rating).take(10)
//      .map(t => (titles(t.product),t.rating))
//    moviesForUserOrder.sortBy(_._1).foreach(println)
//    println("====================================")

    //预测10名的电影
//    predictedRating.map(t=>(titles(t.product),t.rating)).sortBy(_._1).foreach(println)


    //物品推荐相关，计算余弦相似度
//    val aMatrix = new DoubleMatrix(Array(1.0,2.0,3.0))
//    println(aMatrix)

    //物品567的向量，只有一个元素
//    val itemFactor = model.productFeatures.lookup(567).head
//    val itemVector = new DoubleMatrix(itemFactor)
//    val itemCosin = cosineSimilarity(itemVector,itemVector)
//    println(itemCosin)

    //各个物品和567的相似度
//    val sims = model.productFeatures.map{case (id,factor) =>
//        val factorVertor = new DoubleMatrix(factor)
//        val sim = cosineSimilarity(factorVertor,itemVector)
//      (id,sim)
//    }

    //相似度前十的物品
//    val sortedSims = sims.top(11)(Ordering.by[(Int,Double),Double]{
//      case (id,sim) => sim
//    })
//    sortedSims.foreach(println)
//    println("========================")
//    val sortedSims1 = sortedSims.slice(1,11)
//    sortedSims1.foreach(println)
    //计算方差
    //    val actualAndPredictMSE = actualAndPredict.map{
    //      case ((user,item),(act,predict)) => math.pow(act-predict,2.0)
    //    }.reduce(_ + _) /actualAndPredict.count()
    //    println("MSE======"+actualAndPredictMSE)
    //    println(actualAndPredict.first())

    //真实值和预测值的集合
    //    val actualData = rating.map{case Rating(user,item,rating)=>rating}.collect().toSeq;
    //    val predictData = predict.map{case ((user,item),rating) => rating}.collect().toSeq;

    val actualMovies = moviesForUser.map(_.product)
    val predictedMovies = topKRecs.map(_.product)
    val mapk = apk(actualMovies,predictedMovies,10)
    println(mapk)
  }

  //k值平均率
  def apk(actual: Seq[Int],predicted: Seq[Int],k:Int)={
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for((p,i) <- predK.zipWithIndex){
      if (actual.contains(p)){
        numHits += 1.0
        score += numHits/(i.toDouble+1.0)
      }
    }
    if (actual.isEmpty){
      1.0
    }else{
      score / math.min(actual.size,k).toDouble
    }
  }

  //余弦相似度
  def cosineSimilarity(vec1:DoubleMatrix,vec2:DoubleMatrix): Double ={
    vec1.dot(vec2)/(vec1.norm2() * vec2.norm2())
  }
}
