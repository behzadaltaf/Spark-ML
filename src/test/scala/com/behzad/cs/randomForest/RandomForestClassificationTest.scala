package com.behzad.cs.randomForest

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
class RandomForestClassificationTest extends SpecificationWithJUnit with BeforeAfterExample {

  var param: Array[String] = _
  var pcaParam: Array[String] = _

  val inputFileName = "data/Features.csv"
  val pcaInputFileName = "pca/PCA_output"
  val outputFileName = "data/RFC_output"
  val outputModelLocation = "data/RFC_Model"

  val outputFile: File = new File(outputFileName)
  val outputModel: File = new File(outputModelLocation)

  override protected def before {
    if (outputFile.exists()) outputFile.delete()
    if (outputModel.exists()) FileUtils.deleteDirectory(outputModel)
    param = Array(inputFileName, outputFileName, outputModelLocation)
    pcaParam = Array(pcaInputFileName, outputFileName, outputModelLocation)
  }

  "Subject Segmentation Random Forest Classification implementation" should {
    "create an output file and an output model which is a directory with supplied names" in {
      RandomForestClassification.main(param)
      outputFile must exist
      outputFile must beAFile
      outputFile must haveName("RFC_output")
      outputModel must exist
      outputModel must beADirectory
      outputModel must haveName("RFC_Model")
    }
  }


/*
  "Subject Segmentation Random Forest Classification implementation" should {
    "be able to use output from PCA as an input and " +
      "create an output file and an output model which is a directory with supplied names" in {
      RandomForestClassification.main(pcaParam)
      outputFile must exist
      outputFile must beAFile
      outputFile must haveName("SubjectRFC_output")
      outputModel must exist
      outputModel must beADirectory
      outputModel must haveName("SubjectRFC_Model")
    }
  }
*/

  override protected def after: Unit = {
    outputFile.delete()
    FileUtils.deleteDirectory(outputModel)
  }
}
