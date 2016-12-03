package com.behzad.randomForest

import java.io.File

import org.junit.runner.RunWith
import org.specs2.matcher.FileMatchers._
import org.specs2.mutable.SpecificationWithJUnit
import org.specs2.runner.JUnitRunner
import org.specs2.specification.BeforeAfterExample

/**
  * Created by Behzad Altaf
  */
@RunWith(classOf[JUnitRunner])
class RandomForestPredictionTest extends SpecificationWithJUnit with BeforeAfterExample {

  var param: Array[String] = _

  val inputFileForModelGeneration = "data/Features.csv"
  val inputFileForRFAnalysis = "data/randomforest/randomForestAnalysis.csv"
  val outputFileName = "data/randomforest/randomforestAnalysis_output"

  val outputFile: File = new File(outputFileName)

  override protected def before {
    if (outputFile.exists()) outputFile.delete()
    param = Array(inputFileForModelGeneration, inputFileForRFAnalysis, outputFileName)
  }

  "Subject Segmentation Random Forest Prediction implementation" should {
    "create an output file with cluster prediction without saving the model" in {
      RandomForestPrediction.main(param)
      outputFile must exist
      outputFile must beAFile
      outputFile must haveName("randomforestAnalysis_output")
    }
  }

  override protected def after: Unit = {
    outputFile.delete()
  }
}