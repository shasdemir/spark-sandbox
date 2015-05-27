import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}


object RecordLinkage {
    val conf = new SparkConf().setMaster("local[*]").setAppName("RecordLinkage")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


    def main(args: Array[String]): Unit = {
        val dataFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/RecordLinkage/Blocks"
    }
}
