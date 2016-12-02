/**
  *
  * A PCA org.apache.spark.apache.org/docs/latest/mllib-dimensionality-reduction.html#principal-component-analysis-pca
  * implementation for subject segmentation that computes principal components on source vectors and uses them to
  * project the vectors into a low-dimensional space while keeping associated labels.
  *
  * The input file to the program is of the following CSV format
  *
  * |-----------+-------------------------+----------
  * |SubjectID | ... feature vectors ... | SegmentID
  * |-----------+-------------------------+----------
  *
  * In data cleansing step header, the first and the last columns are dropped,
  * The feature vectors can be expanded and collapsed as desired but a minimum one is required.
  *
  * The output of the program is in the following format, this can be directly used for k-means/Random Forest
  * implementations for Subject Segmentation
  *
  * |-----------+-------------------------+----------
  * |SubjectID | ... feature vectors ... | SegmentID
  * |-----------+-------------------------+----------
  *
  * The subject ID and the header columns are added to enable direct k-means/Random Forest usage
  */

package com.behzad.cs.pca

import com.behzad.cs.{AbstractParams, MLUtil}
import MLUtil._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.PCA
import scopt.OptionParser

/**
  * Created by Behzad Altaf
  */
object PCA {

  case class Params(
                     inputFile: String = null,
                     outputFile: String = null,
                     k: Int = 10,
                     master: String = "local",
                     appName: String = "PCA for Subject Segmentation") extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    //Takes care of command line input params and provides in a Params object
    val parser = new OptionParser[Params]("PCA") {
      head("PCA: a Principal Component Analysis on Subject Segmentation data.")
      opt[Int]("k")
        .text(s"min number of principal components, default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[String]("master")
        .text(s"master url, default: ${defaultParams.master}")
        .action((x, c) => c.copy(master = x))
      opt[String]("appName")
        .text(s"application name, default: ${defaultParams.appName}")
        .action((x, c) => c.copy(appName = x))
      arg[String]("inputFile")
        .required()
        .text("input file the subject dataset, required")
        .action((x, c) => c.copy(inputFile = x))
      arg[String]("outputFile")
        .required()
        .text("output file for the run, required")
        .action((x, c) => c.copy(outputFile = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          |The input file is of the following format
          |-----------+-------------------------+----------
          |SubjectID | ... feature vectors ... | SegmentID
          |-----------+-------------------------+----------
          |
          |In data cleansing step header, the first column is dropped. 
          |The feature vectors can be expanded and collapsed as desired but a minimum one is required.
          |
          | bin/spark-submit --class com.wipro.cto.cs.pca.PCA \
          |  SubjectSegmentation-1.0.0-SNAPSHOT.jar \
          |  --k 8 \
          |  --appName Subject_Segmentation PCA \
          |  --master spark://127.0.0.1/master \ 
          |  data/InputData.csv \
          |  data/PCA.out
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {

    //Spark configuration is created
    val conf = new SparkConf().setAppName(params.appName).setMaster(params.master)

    //Spark Context is created
    val sc = new SparkContext(conf)

    //Creates an RDD from the input file
    val rawData = sc.textFile(params.inputFile)

    //Drops the header and the first columns
    val cleansedData = getCleansedData(rawData)

    //Creates PCA using the input k, k < n where n are the features and k is the projection 
    val pca = new PCA(params.k)

    //Creates a PCAMODEL using the features 
    val pcaModel = pca.fit(cleansedData.map(_.features))

    //Projects the final k from the original n
    val projected = cleansedData.map(p => p.copy(features = pcaModel.transform(p.features)))

    //Transforms data to meet the input of k-means/Random Forest
    val transformedData = projected.map(x => s"Nil,${x.features.toArray.mkString(",")},${(x.label + 1.0)}")

    //Creates a header to be supplied
    val fakeHeader = Array.tabulate(params.k + 2)("PlaceHolderHeader " + _).mkString(",")

    //Creates a list from the header
    val list = scala.collection.mutable.MutableList[String]()
    list += fakeHeader

    //Creates an RDD[String] from the header list
    val headerRDD = sc.makeRDD(list)

    //Does a union of the Header RDD and the projected RDD
    val finalRDD = headerRDD.union(transformedData)

    //Saves the RDDs as a single file
    finalRDD.coalesce(1).saveAsTextFile(params.outputFile)

    //Spark Context is stopped
    sc.stop()
  }
}
