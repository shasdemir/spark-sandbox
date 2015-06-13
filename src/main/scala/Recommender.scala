/**
 * Created by sukruhasdemir on 13/06/15.
 */

import org.apache.spark.{SparkContext, SparkConf}

object Recommender {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Recommender").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)

    val dataLocation = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/Recommenders/profiledata_06-May-2005/"
    val userArtistDataFile = dataLocation + "user_artist_data.txt"

    def main(Args: Array[String]): Unit = {
        val rawUserArtistData = sc.textFile(userArtistDataFile)


    }
}
