/**
  * This is a utility class that contains the common methods that are used in the Subject Segmentation project using
  * Spark-MLLIB org.apache.spark.apache.org/docs/latest/mllib-guide.html project.
  *
  */

package com.behzad

import java.util.concurrent.TimeUnit

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
/**
  * Created by Behzad Altaf
  */
object MLUtil {

  /**
    * This method drops the header from a RDD.
    */
  def dropHeaderFromRDD(rawData: RDD[String]): RDD[String] = {
    val rawDataWithoutHeader = rawData.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    rawDataWithoutHeader
  }

  /**
    * This method takes an RDD[String] and converts it into and RDD[LabeledPoint]
    * org.apache.spark.apache.org/docs/latest/mllib-data-types.html#labeled-point
    */
  def getLabeledPointRDD(rawDataWithoutHeader: RDD[String]): RDD[LabeledPoint] = {
    val data = rawDataWithoutHeader.map { line =>
      val values = line.split(',').drop(1).map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
    data
  }

  /**
    * This method takes an RDD[String] and converts it into and RDD[Vector]
    * org.apache.spark.apache.org/docs/latest/mllib-data-types.html#local-vector
    */
  def getFeatureVectorsRDD(rawDataWithoutHeader: RDD[String]): RDD[Vector] = {
    val data = rawDataWithoutHeader.map { line =>
      val values = line.split(',').drop(1).map(_.toDouble)
      val featureVector = Vectors.dense(values)
      featureVector
    }
    data
  }

  /**
    * This method takes an RDD[String] that contains the header, drops it first
    * and converts it into and RDD[LabeledPoint]
    * org.apache.spark.apache.org/docs/latest/mllib-data-types.html#labeled-point
    */
  def getCleansedData(rawData: RDD[String]): RDD[LabeledPoint] = {
    getLabeledPointRDD(dropHeaderFromRDD(rawData))
  }

  /**
    * This method takes an RDD[String] and that contains the header, drops it first
    * and converts it into and RDD[Vector]
    * [org.apache.spark.apache.org/docs/latest/mllib-data-types.html#local-vector]
    */
  def getCleansedDataForAnalysis(rawData: RDD[String]): RDD[Vector] = {
    getFeatureVectorsRDD(dropHeaderFromRDD(rawData))
  }

  /**
    * This method takes a feature Vector [org.apache.spark.apache.org/docs/latest/mllib-data-types.html#local-vector]
    * and adds the prediction to it separated by a comma
    */
  def formatOutputString(featureVector: Vector, prediction: Double): String = {
    (s"${featureVector.toArray.mkString(",")},${prediction.toString()}")
  }

  /**
    * This method takes a long value and converts it into HH:MM:SS:sss string
    */
  def convertLongtoTime(time: Long): String = {
    val hrs = TimeUnit.MILLISECONDS.toHours(time) % 24
    val min = TimeUnit.MILLISECONDS.toMinutes(time) % 60
    val sec = TimeUnit.MILLISECONDS.toSeconds(time) % 60
    val ms = TimeUnit.MILLISECONDS.toMillis(time) % 1000
    (s"HH:MM:SS:sss  $hrs:$min:$sec:$ms")
  }

}