/**
  * A Random Forest classification org.apache.spark.apache.org/docs/latest/mllib-ensembles.html#random-forests
  * implementation that classifies feature vectors into a classes and saves the model for prediction.
  *
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

package com.behzad.randomForest

import com.behzad.{AbstractParams, MLUtil}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import scopt.OptionParser
import com.behzad.MLUtil._

/**
  * Created by Behzad Altaf
  */
object RandomForestClassification {

  case class Params(
                     inputFile: String = null,
                     outputFile: String = null,
                     outputModelLocation: String = null,
                     master: String = "local",
                     appName: String = "Subject Segmentation using Random Forest Classification",
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
      head("SubjectRandomForestCLassification: a random forest classification run on Subject Segmentation data.")
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
          | bin/spark-submit --class com.wipro.cto.cs.randomForest.SubjectRandomForestCLassification \
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
  def writeOutputFile(params: Params, headerString: String, trainingEvaluationMetrics: MulticlassMetrics,
                      testEvaluationMetrics: MulticlassMetrics,
                      counts: (Long, Long, Long), learnedClassificationModel: String, totalTime: String): Unit = {
    val out = new java.io.FileWriter(params.outputFile)
    out.write(s"Total time taken: $totalTime\n\n")
    out.write(s"Headers \n$headerString\n\n")
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
    out.write(s"Learned classification forest model: $learnedClassificationModel")
    out.close()
  }

  /**
    * This method caches or uncaches the RDDs based on the flag provided
    */
  def cacheRDDs(data: RDD[LabeledPoint], trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint],
                cache: Boolean) = {
    if (cache) {
      data.cache()
      trainingData.cache()
      testData.cache
    }
    else {
      data.unpersist()
      trainingData.unpersist()
      testData.unpersist()
    }
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

    //Creates the header map for the output, this is basically a feature selection from the header for
    //Data Scientist analysis of the features used in the Random Forest generation
    val headerMap = rawData.first().split(',').drop(1).dropRight(1).zipWithIndex

    //Formatting the features from the header 
    val headerString = headerMap.map(x => s"feature ${x._2}  => ${x._1}").mkString("\n")

    //Formatting the features from the header creates an Array for formatted feature and index
    val headerSeq = headerMap.map(_.swap).map { case (x, y) => (s"feature $x ", y) }

    //Drops the header, first column and the last column
    val cleansedData = getCleansedData(rawData)

    //Splits the input RDD into test and training
    val splits = cleansedData.randomSplit(Array(params.training, params.test))

    //Naming the splits for named usage
    val (trainingData, testData) = (splits(0), splits(1))

    //Caching the RDDs as will be used multiple times
    cacheRDDs(cleansedData, trainingData, testData, true)

    //Creates the Random Forest model from input parameters
    val model = RandomForest.trainClassifier(trainingData, params.numClasses, params.categoricalFeaturesInfo,
      params.numTrees, params.featureSubsetStrategy, params.impurity, params.maxDepth, params.maxBins)

    //Gets the evaluation matrix from the test RDD 
    val testEvaluationMetrics = getEvaluationMatrix(testData, model)

    //Gets the evaluation matrix from the training RDD
    val trainingEvaluationMetrics = getEvaluationMatrix(trainingData, model)

    //Get counts of the vectors from the Original and the split RDDs
    val counts = getCounts(cleansedData, trainingData, testData)

    //Uncache the RDDs will not be used further
    cacheRDDs(cleansedData, trainingData, testData, false)

    //Create the learned classification model to assist the Data Scientist by replacing the features into named columns
    val learnedClassificationModel = headerSeq.foldLeft(model.toDebugString) {
      case (z, (s, r)) => z.replaceAll(s, s"$r ")
    }

    //Saves the Random Forest Model to the provided location 
    model.save(sc, params.outputModelLocation)

    //Measure time taken by the process
    val totalTime = convertLongtoTime(System.currentTimeMillis() - startTimer)

    //Write to an output file the runtime params and evaluations
    writeOutputFile(params, headerString, trainingEvaluationMetrics, testEvaluationMetrics, counts,
      learnedClassificationModel, totalTime)

    //Spark Context is stopped
    sc.stop()

  }
}