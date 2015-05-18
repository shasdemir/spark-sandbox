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
        val classUdf = udf[Double, String](_.toDouble - 1.0)  // categorical variables start from 0 in MLLib
        val genderUdf = udf[Double, String](gen => if (gen == "male") 1.0 else 0.0)

        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUdf(trainingCSV("Pclass")))
                .withColumn("Gender", genderUdf(trainingCSV("Sex")))
                .select("Survival", "Class", "Gender")
        trainingDataCasted.show()

        val testDataCasted = testCSV
                .withColumn("Class", classUdf(testCSV("Pclass")))
                .withColumn("Gender", genderUdf(testCSV("Sex")))
                .select("Class", "Gender")
        testDataCasted.show()

        val trainingFeatures = trainingDataCasted.map(row =>
            LabeledPoint(row.getInt(0), Vectors.dense(row.getDouble(1), row.getDouble(2))))
        val testFeatures = testDataCasted.map(row => Vector(row.getDouble(1), row.getDouble(2)))

        val splits = trainingFeatures.randomSplit(Array(0.8, 0.2), seed=11L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())
        // *** data prep finishes here ***

        val initialLRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(initialTrainingFeatures)

        val LRValidationResults = validationFeatures.map {
            case LabeledPoint(label, features) => (initialLRModel.predict(features), label)
        }.cache()

        val classificationError = LRValidationResults.filter(tuple => tuple._1 != tuple._2).count().toDouble /
        LRValidationResults.count()

        println("Classification error rate: " + classificationError)
        val validationMetrics = new MulticlassMetrics(LRValidationResults)
        println("Logistic Regression precision: " + validationMetrics.precision)
        println("Logistic Regression recall: " + validationMetrics.recall)


        // evaluate over test data

        // predict test data to submit
    }
}