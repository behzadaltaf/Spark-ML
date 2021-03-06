/**
  * A Random Forest classification org.apache.spark.apache.org/docs/latest/mllib-ensembles.html#random-forests
  * implementation that classifies a feature vector into a class using an already built model from
  * SubjectRandomForestClassification.
  *
  * The input file to the program is of the following CSV format
  *
  * |-----------+----------------------------------
  * |SubjectID , ... feature vectors ...
  * |-----------+----------------------------------
  *
  * The final prediction is saved in the following format without the header
  *
  * |-------------------------+----------
  * | ... feature vectors ... | SegmentID
  * |-------------------------+----------
  *
  */

package com.behzad.randomForest

import com.behzad.{AbstractParams, MLUtil}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.model.RandomForestModel
import scopt.OptionParser
import com.behzad.MLUtil._

/**
  * Created by Behzad Altaf
  */
object RandomForestAnalysis {

  case class Params(
                     inputFile: String = null,
                     outputFile: String = null,
                     modelLocation: String = null,
                     master: String = "local",
                     appName: String = "Subject Segmentation Analysis using Random Forest Classification")
    extends AbstractParams[Params]

  def main(args: Array[String]) = {
    val defaultParams = Params()

    //Takes care of command line input params and provides in a Params object
    val parser = new OptionParser[Params]("RandomForestAnalysis") {
      head("RandomForestAnalysis: a random forest analysis run on Subject Segmentation data.")
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
      arg[String]("modelLocation")
        .required()
        .text("location for the model to be loaded from, required")
        .action((x, c) => c.copy(modelLocation = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          |The input file is of the following CSV format
          |
          |-----------+----------------------------------
          |SubjectID , ... feature vectors ...
          |-----------+----------------------------------
          |
          |In data cleansing step the first column and the header is dropped,           
          |The feature vectors can be expanded and collapsed as desired but a minimum one is required.
          |The feature vectors should be mapped to the same length when the Random Forest Model
          |was generated and saved.
          |
          | bin/spark-submit --class com.wipro.cto.cs.randomForest.RandomForestAnalysis \
          |  SubjectSegmentation-1.0.0-SNAPSHOT.jar \
          |  --appName Subject_Segmentation_RFA \
          |  --master spark://127.0.0.1/master \ 
          |  data/InputData.csv \
          |  data/Subject.out \
          |  model/SubjectRFModel
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
    val cleansedData = getCleansedDataForAnalysis(rawData)

    //Loads the Random Forest model from the provided location
    val model = RandomForestModel.load(sc, params.modelLocation)

    //Predicts the feature vector into a cluster and appends the cluster info in the end
    val predictedData = cleansedData.map { datum =>
      val prediction = model.predict(datum)
      formatOutputString(datum, prediction)
    }

    //Saves the predictions to a provided location 
    predictedData.saveAsTextFile(params.outputFile)

    //Spark Context is stopped
    sc.stop()

  }
}