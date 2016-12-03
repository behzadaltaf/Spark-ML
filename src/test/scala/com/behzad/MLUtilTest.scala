package com.behzad

import org.apache.spark.mllib.linalg.Vectors
import org.junit.runner.RunWith
import org.specs2.mutable.SpecificationWithJUnit
import org.specs2.runner.JUnitRunner

/**
  * Created by Behzad Altaf
  */
@RunWith(classOf[JUnitRunner])
class MLUtilTest extends SpecificationWithJUnit {

  "The convertLongtoTime method in MLUtilTest utility " should {
    "format string of time 1447868114664l as HH:MM:SS:sss  17:35:14:664" in {
      val formattedTime = MLUtil.convertLongtoTime(1447868114664l)
      formattedTime must_== "HH:MM:SS:sss  17:35:14:664"
    }
  }
  
  "The formatOutputString method in MLUtilTest utility " should {
    "concatenate vector and prediction as comma separated" in {
      val vector = Vectors.dense(1.0, 2.0, 3.0)
      val formattedString = MLUtil.formatOutputString(vector, 5.0)
      formattedString must_== "1.0,2.0,3.0,5.0"
    }
  }
}
