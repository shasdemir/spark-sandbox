import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import au.com.bytecode.opencsv.CSVReader
import java.io.StringReader


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

    def loadTrainingData(): RDD[Passenger] = {
        val trainDataText = sc.textFile(trainDataFile)
        val trainDataParsed = trainDataText.map{line =>
            val reader = new CSVReader(new StringReader(line));
            reader.readNext();
        }
        trainDataParsed.cache()

        // println(trainDataParsed.take(10).map(_.mkString(" ")).mkString("\n"))
        val headerlessTrainDataParsed = sc.parallelize(trainDataParsed.take(trainDataParsed.count().toInt).drop(1))

        val trainData = headerlessTrainDataParsed.map(lineArray => new Passenger(
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
        )).cache()

        trainData
    }

    def main(args: Array[String]): Unit = {
        val trainingData = loadTrainingData()
        println(trainingData.take(2).mkString("\n\n"))
    }
}