/**
  * A Random Forest classification org.apache.spark.apache.org/docs/latest/mllib-ensembles.html#random-forests
  * implementation that classifies feature vectors into a classes. This program uses one set of data to create the
  * Random Forest Model and uses that model to run prediction without saving the model.
  *
  * This program is compatible with Spark MLLIB 1.2.1, the save Model methods were introduced into Spark 1.3+
  *
  * The input files to the program are of the following CSV format
  *
  * |-----------+-------------------------+----------
  * |SubjectID | ... feature vectors ... | SegmentID
  * |-----------+-------------------------+----------
  *
  * In data cleansing step header, the first and the last columns are dropped,
  * The feature vectors can be expanded and collapsed as desired but a minimum one is required.
  */

package com.behzad.randomForest

import com.behzad.{AbstractParams, MLUtil}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/**
  * Created by Behzad Altaf
  */
object RandomForestPrediction {

  case class Params(
                     inputFileForModelGeneration: String = null,
                     inputFileForRFAnalysis: String = null,
                     outputFile: String = null,
                     master: String = "local",
                     appName: String = "Subject Segmentation using Random Forest Prediction",
                     numClasses: Int = 6,
                     numTrees: Int = 3,
                     maxDepth: Int = 3,
                     maxBins: Int = 4,
                     training: Double = 0.7,
                     test: Double = 0.3,
                     featureSubsetStrategy: String = "auto",
                     impurity: String = "gini",
                     categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    //Takes care of command line input params and provides in a Params object
    val parser = new OptionParser[Params]("SubjectRandomForestCLassification") {
      head("SubjectRandomForestCLassification: a random forest prediction run on Subject Segmentation data.")
      opt[Int]("numClasses")
        .text(s"number of classes, default: ${defaultParams.numClasses}")
        .action((x, c) => c.copy(numClasses = x))
      opt[Int]("numTrees")
        .text(s"number Of trees , default: ${defaultParams.numTrees}")
        .action((x, c) => c.copy(numTrees = x))
      opt[Int]("maxDepth")
        .text(s"maximum depth of trees , default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("maxBins")
        .text(s"maximum number of bins, default: ${defaultParams.maxBins}")
        .action((x, c) => c.copy(maxBins = x))
      opt[Double]("training")
        .text(s"training slice, default: ${defaultParams.training}")
        .action((x, c) => c.copy(training = x))
      opt[Double]("test")
        .text(s"test slice, default: ${defaultParams.test}")
        .action((x, c) => c.copy(test = x))
      opt[String]("featureSubsetStrategy")
        .text(s"feature Subset Strategy, default: ${defaultParams.featureSubsetStrategy}")
        .action((x, c) => c.copy(featureSubsetStrategy = x))
      opt[String]("impurity")
        .text(s"impurity, default: ${defaultParams.impurity}")
        .action((x, c) => c.copy(impurity = x))
      opt[Map[Int, Int]]("categoricalFeaturesInfo")
        .text(s"categoricalFeaturesInfo k1=v1,k2=v2..., default: ${defaultParams.categoricalFeaturesInfo}")
        .action((x, c) => c.copy(categoricalFeaturesInfo = x))
      opt[String]("master")
        .text(s"master url, default: ${defaultParams.master}")
        .action((x, c) => c.copy(master = x))
      opt[String]("appName")
        .text(s"application name, default: ${defaultParams.appName}")
        .action((x, c) => c.copy(appName = x))
      arg[String]("inputFileForModelGeneration")
        .required()
        .text("input file the model generation, required")
        .action((x, c) => c.copy(inputFileForModelGeneration = x))
      arg[String]("inputFileForRFAnalysis")
        .required()
        .text("input file the analysis, required")
        .action((x, c) => c.copy(inputFileForRFAnalysis = x))
      arg[String]("outputFile")
        .required()
        .text("output file for the run, required")
        .action((x, c) => c.copy(outputFile = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |          
          |The input files are of the following format
          |-----------+-------------------------+----------
          |SubjectID | ... feature vectors ... | SegmentID
          |-----------+-------------------------+----------
          |
          |In data cleansing step header, the first and the last columns are dropped, 
          |The feature vectors can be expanded and collapsed as desired but a minimum one is required.
          |
          | bin/spark-submit --class com.wipro.cto.cs.randomForest.RandomForestPrediction \
          |  SubjectSegmentation-1.0.0-SNAPSHOT.jar \
          |  --numClasses 6 \
          |  --numTrees 30 \
          |  --maxDepth 30 \
          |  --maxBins 20 \
          |  --training 0.8
          |  --test 0.2\
          |  --featureSubsetStrategy auto \
          |  --impurity gini\
          |  --categoricalFeaturesInfo 1=2,4=3 \
          |  --appName Subject_Segmentation_RF \
          |  --master spark://127.0.0.1/master \ 
          |  data/InputDataForModelGeneration.csv \
          |  data/InputDataForRFAnalysis.csv \
          |  data/SubjectRF.out
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
  def getEvaluationMatrix(labeledPoints: RDD[LabeledPoint], model: RandomForestModel): MulticlassMetrics = {
    val labelAndPreds = labeledPoints.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    new MulticlassMetrics(labelAndPreds)
  }

  /**
    * This method gets the counts of the feature vectors from the provided RDDs
    */
  def getCounts(total: RDD[LabeledPoint], training: RDD[LabeledPoint], test: RDD[LabeledPoint]): (Long, Long, Long) = {
    (total.count(), training.count(), test.count())
  }

  /**
    * This method writes an output file containing the run time params and output
    */
  def writeOutputFile(params: Params, trainingEvaluationMetrics: MulticlassMetrics,
                      testEvaluationMetrics: MulticlassMetrics,
                      counts: (Long, Long, Long), totalTimeForModelGeneration: String,
                      rFAnalysisEvaluationMetrics: MulticlassMetrics, totalTimeForRFAnalysis: String): Unit = {
    val out = new java.io.FileWriter(params.outputFile)
    out.write(s"Total time taken for model generation: $totalTimeForModelGeneration\n\n")
    out.write(s"Total time taken for RF analysis: $totalTimeForRFAnalysis\n\n")
    out.write(s"Input Parameters: $params\n\n")
    out.write(s"Training Run precision: ${trainingEvaluationMetrics.precision}\n\n")
    out.write("Training Confusion Matrix:\n")
    out.write(trainingEvaluationMetrics.labels.map(_.toString()).mkString("\t"))
    out.write("\n\n")
    out.write(trainingEvaluationMetrics.confusionMatrix.toString())
    out.write("\n\n")
    out.write(s"Test Run precision: ${testEvaluationMetrics.precision} \n\n")
    out.write("Test Confusion Matrix:\n")
    out.write(testEvaluationMetrics.labels.map(_.toString()).mkString("\t"))
    out.write("\n\n")
    out.write(testEvaluationMetrics.confusionMatrix.toString)
    out.write("\n\n")
    out.write(s"Counts => Total: ${counts._1} Training: ${counts._2} Test: ${counts._3}\n\n")
    out.write(s"Difference: ${(counts._1 - (counts._2 + counts._3))}\n\n")
    out.write("\n\n===============================================================\n\n")
    out.write(s"RF Analysis Run precision: ${rFAnalysisEvaluationMetrics.precision}\n\n")
    out.write("RF Analysis Confusion Matrix:\n")
    out.write(rFAnalysisEvaluationMetrics.labels.map(_.toString()).mkString("\t"))
    out.write("\n\n")
    out.write(rFAnalysisEvaluationMetrics.confusionMatrix.toString())
    out.close()
  }

  def run(params: Params): Unit = {

    //Start time to measure program performance of Model Generation
    val startTimerForModelGeneration = System.currentTimeMillis()

    //Spark configuration is created
    val conf = new SparkConf().setAppName(params.appName).setMaster(params.master)

    //Spark Context is created
    val sc = new SparkContext(conf)

    //Creates an RDD from the input file
    val rawData = sc.textFile(params.inputFileForModelGeneration)

    //Drops the header, first column and the last column
    val cleansedData = getCleansedData(rawData)

    //Splits the input RDD into test and training
    val splits = cleansedData.randomSplit(Array(params.training, params.test))

    //Naming the splits for named usage
    val (trainingData, testData) = (splits(0), splits(1))

    //Creates the Random Forest model from input parameters
    val model = RandomForest.trainClassifier(trainingData, params.numClasses, params.categoricalFeaturesInfo,
      params.numTrees, params.featureSubsetStrategy, params.impurity, params.maxDepth, params.maxBins)

    //Gets the evaluation matrix from the test RDD 
    val testEvaluationMetrics = getEvaluationMatrix(testData, model)

    //Gets the evaluation matrix from the training RDD
    val trainingEvaluationMetrics = getEvaluationMatrix(trainingData, model)

    //Get counts of the vectors from the Original and the split RDDs
    val counts = getCounts(cleansedData, trainingData, testData)

    //Calculate the total time taken for model generation
    val totalTimeForModelGeneration = convertLongtoTime(System.currentTimeMillis() - startTimerForModelGeneration)

    //Start time to measure program performance of Random Forest Analysis
    val startTimerForRFAnalysis = System.currentTimeMillis()

    //Creates an RDD from the input file
    val rawAnalysisData = sc.textFile(params.inputFileForRFAnalysis)

    //Drops the header, first column and the last column
    val cleansedRFAnalysisData = getCleansedData(rawAnalysisData)

    //Gets the evaluation matrix from the RF RDD
    val rFAnalysisEvaluationMetrics = getEvaluationMatrix(cleansedRFAnalysisData, model)

    //Calculate the total time taken for Data analysis using the created model
    val totalTimeForRFAnalysis = convertLongtoTime(System.currentTimeMillis() - startTimerForRFAnalysis)

    //Write to an output file the runtime params and evaluations
    writeOutputFile(params, trainingEvaluationMetrics, testEvaluationMetrics, counts, totalTimeForModelGeneration,
      rFAnalysisEvaluationMetrics, totalTimeForRFAnalysis)

    //Spark Context is stopped
    sc.stop()

  }
}