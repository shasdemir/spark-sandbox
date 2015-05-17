import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import au.com.bytecode.opencsv.CSVReader
import java.io.StringReader
import scala.io.Source


object Titanic {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Titanic")
    val sc = new SparkContext(conf)

    val dataFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/titanic/"
    val trainDataFile = dataFolder + "train.csv"
    val testDataFile = dataFolder + "test.csv"

    case class Passenger(PassengerId: Integer, Survived: Option[Integer], Pclass: Option[Integer],
                         Name: Option[String], Sex: Option[String], Age: Option[Double], SibSp: Option[Integer],
                         Parch: Option[Integer], Ticket: Option[String], Fare: Option[Double],
                         Cabin: Option[String], Embarked: Option[String])

    def loadDataFile(path: String): RDD[Passenger] = {
        val dataText = Source.fromFile(path).getLines().toArray.tail  //remove headers
        val dataParsed = dataText.map{line =>
                val reader = new CSVReader(new StringReader(line))
                reader.readNext()}

        val dataPassengers = dataParsed.map(lineArray => new Passenger(
            PassengerId = lineArray(0).toInt,
            Survived = if (lineArray(1) != "") Some(lineArray(1).toInt) else None,
            Pclass = if (lineArray(2) != "") Some(lineArray(2).toInt) else None,
            Name = if (lineArray(3) != "") Some(lineArray(3).toString) else None,
            Sex = if (lineArray(4) != "") Some(lineArray(4).toString) else None,
            Age = if (lineArray(5) != "") Some(lineArray(5).toDouble) else None,
            SibSp = if (lineArray(6) != "") Some(lineArray(6).toInt) else None,
            Parch = if (lineArray(7) != "") Some(lineArray(7).toInt) else None,
            Ticket = if (lineArray(8) != "") Some(lineArray(8).toString) else None,
            Fare = if (lineArray(9) != "") Some(lineArray(9).toDouble) else None,
            Cabin = if (lineArray(10) != "") Some(lineArray(10).toString) else None,
            Embarked = if (lineArray(11) != "") Some(lineArray(11).toString) else None
        ))

        sc.parallelize(dataPassengers).cache()
    }

    def main(args: Array[String]): Unit = {
        val trainingData = loadDataFile(trainDataFile)
        val trainingDataCount = trainingData.count()

        println()
        // println(trainingData.take(10).mkString("\n\n"))

        // do every passenger has PassengerId, Survived, Pclass, Sex, Age attributes?
        println("Number of passangers without a Survived attribute: " + trainingData.filter(_.Survived == None).count())
        println("Number of passangers without a Pclass attribute: " + trainingData.filter(_.Pclass == None).count())
        println("Number of passangers without a Sex attribute: " + trainingData.filter(_.Sex == None).count())
        println("Number of passangers without a Age attribute: " + trainingData.filter(_.Age == None).count())  // 177




    }
}