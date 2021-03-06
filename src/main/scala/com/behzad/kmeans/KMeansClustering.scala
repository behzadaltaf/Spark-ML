/**
  * A k-means clustering spark.apache.org/docs/latest/mllib-clustering.html#k-means implementation
  * that clusters a feature vectors into a clusters and saves the model for prediction
  * The input file to the program is of the following CSV format
  *
  * |-----------+-------------------------+----------
  * |SubjectID | ... feature vectors ... | SegmentID
  * |-----------+-------------------------+----------
  *
  * In data cleansing step header, the first and the last columns are dropped,
  * The feature vectors can be expanded and collapsed as desired but a minimum one is required.
  * The model is saved to a location which can be loaded to predict feature vectors.
  *
  */

package com.behzad.kmeans

import com.behzad.{AbstractParams, MLUtil}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import com.behzad.MLUtil._

/**
  * Created by Behzad Altaf
  */
object KMeansClustering {

  object InitializationMode extends Enumeration {
    type InitializationMode = Value
    val Random, Parallel = Value
  }

  import InitializationMode._

  case class Params(
                     inputFile: String = null,
                     outputFile: String = null,
                     outputModelLocation: String = null,
                     initializationMode: InitializationMode = Parallel,
                     seed: Long = scala.util.Random.nextLong(),
                     master: String = "local",
                     appName: String = "Subject Segmentation using k-means",
                     runs: Int = 10,
                     epsilon: Double = 1.0e-6,
                     k: Int = 6) extends AbstractParams[Params]

  def main(args: Array[String]) {

    val defaultParams = Params()

    //Takes care of command line input params and provides in a Params object
    val parser = new OptionParser[Params]("KMeansClustering") {
      head("KMeansClustering: a k-means clustering run on Subject Segmentation data.")
      opt[Int]("k")
        .text(s"min number of k-means partitions, default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("runs")
        .text(s"number Of Runs , default: ${defaultParams.runs}")
        .action((x, c) => c.copy(runs = x))
      opt[Double]("epsilon")
        .text(s"error margin epsilon, default: ${defaultParams.epsilon}")
        .action((x, c) => c.copy(epsilon = x))
      opt[String]("initializationMode")
        .text(s"initialization mode (${InitializationMode.values.mkString(",")}), " +
          s"default: ${defaultParams.initializationMode}")
        .action((x, c) => c.copy(initializationMode = InitializationMode.withName(x)))
      opt[Long]("seed")
        .text(s"seed, default: Randomly generated by scala.util.Random.nextLong()")
        .action((x, c) => c.copy(seed = x))
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
      arg[String]("outputModelLocation")
        .required()
        .text("output location for the model to be saved, required")
        .action((x, c) => c.copy(outputModelLocation = x))

      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          |The input file is of the following format
          |-----------+-------------------------+----------
          |SubjectID | ... feature vectors ... | SegmentID
          |-----------+-------------------------+----------
          |
          |In data cleansing step header, the first and the last columns are dropped, 
          |The feature vectors can be expanded and collapsed as desired but a minimum one is required.
          |
          | bin/spark-submit --class com.wipro.cto.cs.kmeans.KMeansClustering \
          |  SubjectSegmentation-1.0.0-SNAPSHOT.jar \
          |  --k 5 \
          |  --runs 8 \
          |  --epsilon 1.0e-4 \
          |  --initializationMode Random \
          |  --seed 1000 \
          |  --appName Subject_Segmentation_K-Means\
          |  --master spark://127.0.0.1/master \ 
          |  data/InputData.csv \
          |  data/Subjectkmeans.out \
          |  data/SubjectkmeansModel
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  /**
    * This method creates a MulticlassMetrics from LabeledPoints using predictions
    */
  def getEvaluationMatrix(labeledPoints: RDD[LabeledPoint], model: KMeansModel): MulticlassMetrics = {
    val labelAndPreds = labeledPoints.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction.toDouble)
    }
    new MulticlassMetrics(labelAndPreds)
  }

  /**
    * This method writes an output file containing the run time params and output
    */
  def writeOutputFile(params: Params, cost: Double, dataEvaluationMetrics: MulticlassMetrics, totalTime: String): Unit = {
    val out = new java.io.FileWriter(params.outputFile)
    out.write(s"Total time taken: $totalTime\n\n")
    out.write(s"Input Parameters: $params\n\n")
    out.write(s"Total cost = $cost \n")
    out.write(s"Training Run precision: ${dataEvaluationMetrics.precision}\n\n")
    out.write("Training Confusion Matrix: \n")
    out.write(dataEvaluationMetrics.labels.map(_.toString()).mkString("\t"))
    out.write("\n\n")
    out.write(dataEvaluationMetrics.confusionMatrix.toString())
    out.write("\n\n")
    out.close()
  }

  def run(params: Params): Unit = {

    //Start time to measure program performance
    val startTimer = System.currentTimeMillis()

    //Spark configuration is created
    val conf = new SparkConf().setAppName(params.appName).setMaster(params.master)

    //Spark Context is created
    val sc = new SparkContext(conf)

    //Creates an RDD from the input file
    val rawData = sc.textFile(params.inputFile)

    //Drops the header, first column and the last column
    val cleansedData = getCleansedData(rawData)

    //Extracts feature vectors from labeledpoint
    val featureVectors = cleansedData.map(_.features)

    //Determine the input initialization mode
    val initMode = params.initializationMode match {
      case Random => KMeans.RANDOM
      case Parallel => KMeans.K_MEANS_PARALLEL
    }

    //Set the k-means parameters
    val kmeans = new KMeans()
      .setK(params.k)
      .setRuns(params.runs)
      .setEpsilon(params.epsilon)
      .setInitializationMode(initMode)
      .setSeed(params.seed)

    //Create the model 
    val model = kmeans.run(featureVectors)

    //Compute the cost from the generated model and feature vectors
    val cost = model.computeCost(featureVectors)

    //Get the evaluation matrix from the data and the model by running the feature vectors 
    //through the model and finding predictions
    val dataEvaluationMetrics = getEvaluationMatrix(cleansedData, model)

    //Save the model for future usage for prediction
    model.save(sc, params.outputModelLocation)

    val totalTime = convertLongtoTime(System.currentTimeMillis() - startTimer)

    //Write program data to an output file
    writeOutputFile(params, cost, dataEvaluationMetrics, totalTime)

    //Spark Context is stopped
    sc.stop()

  }

}
