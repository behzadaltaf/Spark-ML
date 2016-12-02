package com.behzad.cs.kmeans

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
class KMeansAnalysisTest extends SpecificationWithJUnit with BeforeAfterExample {

  var param: Array[String] = _

  val inputFileName = "data/kmeans/kmeansAnalysis.csv"
  val outputFileName = "data/kmeans/kmeansAnalysis_output"
  val modelLocation = "data/kmeans/KMeansAnalysis_Model"

  val outputFile: File = new File(outputFileName)

  override protected def before {
    if (outputFile.exists()) FileUtils.deleteDirectory(outputFile)
    param = Array(inputFileName, outputFileName, modelLocation)
  }

  "Subject Segmentation k-means Analysis implementation" should {
    "create an output file with predictions using an input file with feature vectors and an existing k-means model" in {
      KMeansAnalysis.main(param)
      outputFile must exist
      outputFile must beADirectory
      outputFile must haveName("kmeansAnalysis_output")
    }
  }

  override protected def after: Unit = {
    FileUtils.deleteDirectory(outputFile)
  }
}