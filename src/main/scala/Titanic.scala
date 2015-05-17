import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.databricks.spark.csv._


object Titanic {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Titanic")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val dataFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/titanic/"
    val trainDataFile = dataFolder + "train.csv"
    val testDataFile = dataFolder + "test.csv"

    def main(args: Array[String]): Unit = {
        val trainingCSV = sqlContext.csvFile(trainDataFile, useHeader = true).cache()
        val testCSV = sqlContext.csvFile(testDataFile, useHeader = true).cache()

        println("Number of passangers without a Survived attribute: " + trainingCSV.filter(trainingCSV("Survived") === "").count())
        println("Number of passangers without a Pclass attribute: " + trainingCSV.filter(trainingCSV("Pclass") === "").count())
        println("Number of passangers without a Sex attribute: " + trainingCSV.filter(trainingCSV("Sex") === "").count())
        println("Number of passangers without a Age attribute: " + trainingCSV.filter(trainingCSV("Age") === "").count())  // 177

        val toInt = udf[Int, String](_.toInt)
        val classUdf = udf[Int, String](_.toInt - 1)  // categorical variables start from 0 in MLLib
        val genderUdf = udf[Int, String](gen => if (gen == "male") 1 else 0)

        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUdf(trainingCSV("Pclass")))
                .withColumn("Gender", genderUdf(trainingCSV("Sex")))
                .select("Survival", "Id", "Class", "Gender")
        trainingDataCasted.show()

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUdf(testCSV("Pclass")))
                .withColumn("Gender", genderUdf(testCSV("Sex")))
                .select("Id", "Class", "Gender")
        testDataCasted.show()


        // prepare features, using only class and gender now
//        def PclassFeatureize(Pclass: Int) = Pclass - 1.0  // categorical variables start from 0 in MLLib
//        def genderFeatureize(gender: String) = if (gender == "male") 1.0 else 0.0
//
//        val trainingData = trainingDataRaw.map(person =>
//            LabeledPoint(person.Survived.get,
//                Vectors.dense(PclassFeatureize(person.Pclass.get), genderFeatureize(person.Sex.get)))).cache()
//
//        val testData = testDataRaw.map(person =>
//            Vectors.dense(PclassFeatureize(person.Pclass.get), genderFeatureize(person.Sex.get))).cache()
//
//        // separate training data to initial training and validation sets
//        val splits = trainingData.randomSplit(Array(0.8, 0.2), seed = 11L)
//        val (initialTrainingData, validationData) = (splits(0).cache(), splits(1).cache())
//        // *** data prep finishes here ***
//
//        val initialLRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(initialTrainingData)
//
//        val LRValidationResults = validationData.map {
//            case LabeledPoint(label, features) => (initialLRModel.predict(features), label)
//        }.cache()
//
//        //val classificationError = LRValidationResults.filter(case (key: Double, value: Double) => key == value).
//        val validationMetrics = new MulticlassMetrics(LRValidationResults)
//        println("Logistic Regression precision: " + validationMetrics.precision)
//        println("Logistic Regression recall: " + validationMetrics.recall)
//
//        println("Type of LRValidationResults: " + LRValidationResults.getClass)



        // try class and gender based LR model
        //val  LRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)

        // evaluate over test data

        // predict test data to submit
    }
}