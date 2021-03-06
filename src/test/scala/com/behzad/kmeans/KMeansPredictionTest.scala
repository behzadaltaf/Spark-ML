package com.behzad.kmeans

import java.io.File

import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.specs2.matcher.FileMatchers._
import org.specs2.mutable.SpecificationWithJUnit
import org.specs2.runner.JUnitRunner
import org.specs2.specification.BeforeAfterExample

/**
  * Created by Behzad Altaf
  */
@RunWith(classOf[JUnitRunner])
class KMeansPredictionTest extends SpecificationWithJUnit with BeforeAfterExample {

  var param: Array[String] = _

  val inputFileName = "data/kmeans/kmeansAnalysis.csv"
  val outputFileName = "data/kmeans/kmeansAnalysis_output"

  val outputFile: File = new File(outputFileName)

  override protected def before {
    if (outputFile.exists()) FileUtils.deleteDirectory(outputFile)
    param = Array(inputFileName, outputFileName)
  }

  "Subject Segmentation k-means clustering implementation" should {
    "create an output file with cluster prediction without saving the model" in {
      KMeansPrediction.main(param)
      outputFile must exist
      outputFile must beADirectory
      outputFile must haveName("kmeansAnalysis_output")
    }
  }

  override protected def after: Unit = {
    FileUtils.deleteDirectory(outputFile)
  }
}
