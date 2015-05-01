import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel


object miniSparkTry {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setMaster("local").setAppName("Mini Spark Try")
        val sc = new SparkContext(conf)

        val inputFile = "/Users/sukruhasdemir/spark-1.2.0/README.md"
        val outputFile = "minitryOutput"

        val input = sc.textFile(inputFile)
        val words = input.flatMap(_.split(" "))
        val counts = words.map((_, 1)).reduceByKey((x, y) => x + y)

        counts.persist(StorageLevel.MEMORY_AND_DISK_2)

        counts.saveAsTextFile(outputFile)

        sc.stop()
    }
}