package dimensionalityReduction
package scala.book降维

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import breeze.linalg.DenseMatrix

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by yuch on 2016/5/8.
  * spark机器学习，分类
  */
object FeatureExtraction {

  def PCA(): Unit ={
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[3]").setAppName("test")
    val sc = new SparkContext(conf)
    //    val path = "D:\\迅雷下载\\lfw-a\\lfw\\*"
    val path = "hdfs://192.168.1.90:8020/tmp/yuch/lfw-a/lfw/*"
    val rdd = sc.wholeTextFiles(path)
    //    print(rdd.first())

    //D:\\迅雷下载\\lfw-a\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg
    val files = rdd.map{ case (fileName,content) =>
      fileName.replace("hdfs://192.168.1.90:8020/tmp/yuch","D:/迅雷下载").replace("/","\\\\")
    }

    println("count==========="+files.count())

    val pixels = files.map{ f =>
      extractPixels(f,50,50)}

    //    println(pixels.take(10).map(_.take(10).mkString("",",",",....")).mkString("\n"))

    //生成向量
    val vectors = pixels.map(p => Vectors.dense(p))
    //    println(vectors.first())

    vectors.setName("image-vertors")
    vectors.cache()

    //正则化
    val scaler = new StandardScaler(true,false).fit(vectors)

    val scaledVectors = vectors.map(v => scaler.transform(v))
    //    println("scaledVectors====="+scaledVectors.count())

    val matrix = new RowMatrix(scaledVectors)
    val matrixrows = matrix.numRows
    val matrixcols = matrix.numCols
    println("matrix.numRows========="+matrixrows)
    println("matrix.numCols========="+matrixcols)
    val k = 10

    //主成分
    val pc = matrix.computePrincipalComponents(k)
    val rows = pc.numRows
    val cols = pc.numCols
    //    println("pc.numRows========="+rows)
    //    println("pc.numCols========="+cols)

    //    val pcBreeze = new DenseMatrix(rows,cols)

    //将原来的照片，投影到主成分上
    val projected = matrix.multiply(pc)
    //    println(projected.numRows(),projected.numCols())

    //    val a = loadImageFromFile("hdfs://192.168.1.100:8020/tmp/MROozieTest/lfw-a/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    //    val aImage = loadImageFromFile("D:\\迅雷下载\\lfw-a\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg")
    //    val grayImage = processImage(aImage,100,100)
    //    ImageIO.write(grayImage,"jpg",new File("D:\\迅雷下载\\lfw-a\\test.jpg"))

    val svd = matrix.computeSVD(10,true)
    val u = svd.U
    val s = svd.s
    val v = svd.V
    println("u.numRows===="+u.numRows(),"u.numCols===="+u.numCols)
    println("s.size===="+s.size)
    println("v.numRows===="+v.numRows,"v.numCols===="+v.numCols)


  }

  //接受一个图片文件位置和需要处理的宽和高，返回包括像素元素的Array[double]
  def extractPixels(path:String,width:Int,height:Int)={
    val raw = loadImageFromFile(path)
    val processed = processImage(raw,width,height)
    getPixelsFromImage(processed)
  }

  //提取纯灰度像素数据作为特征，打平二维的像素矩阵来构造一维的向量
  def getPixelsFromImage(image:BufferedImage) : Array[Double]={
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width*height)
    image.getData.getPixels(0,0,width,height,pixels)
  }

  //完成灰度转化和尺寸改变
  def processImage(image: BufferedImage,width:Int,height:Int):BufferedImage ={
    val bwImage = new BufferedImage(width,height,BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics
    g.drawImage(image,0,0,width,height,null)
    g.dispose()
    bwImage
  }

  //存储图片数据
  def loadImageFromFile(path:String):BufferedImage={
    ImageIO.read(new File(path))
  }

  def main(args: Array[String]): Unit = {
    PCA
  }
}
