package com.behzad

import org.apache.spark.{SparkConf, SparkContext}
import org.specs2.mutable.SpecificationWithJUnit
import org.specs2.specification.{BeforeAfterExample, Fragments, Step}

/**
  * Created by Behzad Altaf
  */
abstract class SparkJobSpec extends SpecificationWithJUnit with BeforeAfterExample {

  @transient var sc: SparkContext = _

  def beforeAll = {
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("test")
    //sc = new SparkContext(conf)
  }

  def afterAll = {
    if (sc != null) {
      sc.stop()
      sc = null
      System.clearProperty("spark.driver.port")
      System.clearProperty("spark.hostPort")
    }
  }

  override def map(fs: => Fragments) = Step(beforeAll) ^ super.map(fs) ^ Step(afterAll)

}


