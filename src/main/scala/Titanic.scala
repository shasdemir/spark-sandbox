import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.sql.{DataFrame, SQLContext, Row}
import org.apache.spark.sql.functions._
import com.databricks.spark.csv._


object FilePaths {
    val mainFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/"
    val dataFolder = mainFolder + "data/titanic/"
    val trainDataFile = dataFolder + "train.csv"
    val testDataFile = dataFolder + "test.csv"
    val resultsFolder = mainFolder + "results/"
}


object TitanicUDFs {
    val toInt = udf[Int, String](_.toInt)
    val classUDF = udf[Double, String](_.toDouble - 1.0)  // categorical variables start from 0 in MLLib
    val genderUDF = udf[Double, String](gen => if (gen == "male") 1.0 else 0.0)
    val ageUDF = udf[Double, String](rawAge => if (rawAge == "") 0.0 else rawAge.toDouble)
}


case class TitanicResult(PassengerId: Int, Survived: Int)


object Titanic {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Titanic")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    type SparkVector = org.apache.spark.mllib.linalg.Vector

    import FilePaths._
    val (trainingCSV, testCSV) = loadTrainingAndTestData()

    import TitanicUDFs._


    def loadTrainingAndTestData(): (DataFrame, DataFrame) = {
        val trainingCSV = sqlContext.csvFile(trainDataFile, useHeader = true).cache()
        val testCSV = sqlContext.csvFile(testDataFile, useHeader = true).cache()
        (trainingCSV, testCSV)
    }


    def prepGenderClassData(): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], RDD[(Int, SparkVector)]) = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .select("Id", "Survival", "Class", "Gender")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .select("Id", "Class", "Gender")

        val trainingFeatures = trainingDataCasted.map(row =>
            LabeledPoint(row.getInt(1), Vectors.dense(row.getDouble(2), row.getDouble(3)))).cache() // label, features
        val testFeatures = testDataCasted.map(row => (row.getInt(0), Vectors.dense(row.getDouble(1), row.getDouble(2)))) // id, features

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassAgeData() = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("AgeMash", ageUDF(trainingCSV("Age")))
                .select("Id", "Survival", "Class", "Gender", "AgeMash").cache()

        val trainingAgeAverages = trainingDataCasted.groupBy("Class", "Gender").avg("AgeMash").rdd.collect()

        def getAverageAge(ageAveragesData: Array[Row], Class: Double, Gender: Double): Double =  // to impute for age
            ageAveragesData.filter(row => row.getDouble(0) == Class && row.getDouble(1) == Gender)(0).getDouble(2)

        val trainingDataImputed = trainingDataCasted.rdd.map { row =>
            Array(row.getInt(0), row.getInt(1), row.getDouble(2), row.getDouble(3),
                if (row.getDouble(4) == 0.0)
                    getAverageAge(trainingAgeAverages, row.getDouble(2), row.getDouble(3))
                else
                    row.getDouble(4)
            )
        }

        trainingDataImputed
    }


    def runGenderClassLRModel(): Unit = {
        val (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures) = prepGenderClassData()

        val initialLRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(initialTrainingFeatures)
        println("initialLRModel weights: " + initialLRModel.weights + initialLRModel.intercept)

        val LRValidationResults = validationFeatures.map {
            case LabeledPoint(label, features) => (initialLRModel.predict(features), label)
        }.cache()

        val classificationError = LRValidationResults.filter(tuple => tuple._1 != tuple._2).count().toDouble /
                LRValidationResults.count()

        println("Logistic Regression classification error rate: " + classificationError)
        val validationMetrics = new MulticlassMetrics(LRValidationResults)
        println("Logistic Regression precision: " + validationMetrics.precision)
        println("Logistic Regression recall: " + validationMetrics.recall)

        // train full model
        val LRGenderClassModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(trainingFeatures)

        // evaluate over test data
        val LRGenderClassResults = testFeatures.map {
            case (idInt, fVector) => (idInt, LRGenderClassModel.predict(fVector).toInt)
        }.cache()

        println(LRGenderClassResults.count())
        println(LRGenderClassResults.take(10).mkString("\n"))
        println("LRGenderClassModel weights: " + LRGenderClassModel.weights + LRGenderClassModel.intercept)

        // save results to csv
        val LRGCResultDF = LRGenderClassResults.map(tuple => new TitanicResult(tuple._1, tuple._2)).toDF()
        LRGCResultDF.show()
        //LRGCResultDF.saveAsCsvFile(resultsFolder + "LRGenderClassResults")  // to submit to kaggle: merge files and header row
    }


    def main(args: Array[String]): Unit = {
        runGenderClassLRModel()

    }
}
