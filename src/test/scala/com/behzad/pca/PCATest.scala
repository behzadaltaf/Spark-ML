package com.behzad.pca

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
class PCATest extends SpecificationWithJUnit with BeforeAfterExample {

  var param: Array[String] = _

  val inputFileName = "data/Features.csv"
  val outputFileName = "data/SubjectPCA_output"

  var outputFile: File = new File(outputFileName)

  override protected def before {
    if (outputFile.exists()) FileUtils.deleteDirectory(outputFile)
    param = Array(inputFileName, outputFileName)
  }

  "Subject Segmentation PCA implementation" should {
    "create an output which is a directory with the supplied name" in {
      PCA.main(param)
      outputFile must exist
      outputFile must beADirectory
      outputFile must haveName("SubjectPCA_output")
    }
  }

  override protected def after: Unit = {
    FileUtils.deleteDirectory(outputFile)
  }
}