/**
 * Created by sukruhasdemir on 13/06/15.
 */
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD

object Recommender {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Recommender").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)

    val dataLocation = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/Recommenders/profiledata_06-May-2005/"
    val userArtistDataFile = dataLocation + "user_artist_data.txt"
    val artistDataFile = dataLocation + "artist_data.txt"
    val artistAliasFile = dataLocation + "artist_alias.txt"

    def main(Args: Array[String]): Unit = {
        val rawUserArtistData = sc.textFile(userArtistDataFile)
        // rawUserArtistData.map(_.split(" ")(0).toDouble).stats

        val rawArtistData = sc.textFile(artistDataFile)

        val artistByID = rawArtistData.flatMap { line =>
            val (id, name) = line.span(_ != '\t')
            if (name.isEmpty) {
                None
            } else {
                try {
                    Some((id.toInt, name.trim))
                } catch {
                    case e: NumberFormatException => None
                }
            }
        }

        val rawArtistAlias = sc.textFile(artistAliasFile)

        val artistAlias = rawArtistAlias.flatMap { line =>
            val tokens = line.split('\t')
            if (tokens(0).isEmpty)
                None
            else
                Some((tokens(0).toInt, tokens(1).toInt))
        }.collectAsMap()
    }
}
