import java.io.StringReader

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import au.com.bytecode.opencsv.CSVReader

object Titanic {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setMaster("local[*]").setAppName("Titanic")
        val sc = new SparkContext(conf)

        val dataFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/titanic/"
        val trainDataFile = dataFolder + "train.csv"
        val testDataFile = dataFolder + "test.csv"

        val trainDataText = sc.textFile(trainDataFile)
        val trainDataParsed = trainDataText.map{line =>
            val reader = new CSVReader(new StringReader(line));
            reader.readNext();
        }

        //println(trainDataParsed.take(3).mkString(" "))
        println(trainDataParsed.count())
        println(trainDataParsed.getClass)
        println(trainDataParsed.take(3)(0).mkString(" "))
    }
}