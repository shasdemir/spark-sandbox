/**
 * Created by sukruhasdemir on 13/06/15.
 */
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation._

object Recommender {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Recommender").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)

    val dataLocation = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/Recommenders/profiledata_06-May-2005/"
    val userArtistDataFile = dataLocation + "user_artist_data.txt"
    val artistDataFile = dataLocation + "artist_data.txt"
    val artistAliasFile = dataLocation + "artist_alias.txt"

    def importData(): (RDD[String], RDD[(Int, String)], scala.collection.Map[Int, Int]) = {  // whatever
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

        (rawUserArtistData, artistByID, artistAlias)
    }

    def spotCheckUser(userid: Int = 2093760, model: MatrixFactorizationModel, rawUserArtistData: RDD[String],
                      artistByID: RDD[(Int, String)]): Unit = {

        val rawArtistsForUser = rawUserArtistData
                .map(_.split(" "))
                .filter { case Array(user, _, _) => user.toInt == userid }

        val existingProducts = rawArtistsForUser
                .map { case Array(_, artist, _) => artist.toInt }
                .collect().toSet

        println("User " + userid + " interacted with items:")
        artistByID.filter { case (id, name) => existingProducts.contains(id) }.values.collect().foreach(println)

        val recommendations = model.recommendProducts(2093760, 5)
        recommendations foreach println

        val recommendedProductIDs = recommendations.map(_.product).toSet

        artistByID.filter { case (id, name) => recommendedProductIDs.contains(id) }.values.collect().foreach(println)
    }

    def main(Args: Array[String]): Unit = {
        val (rawUserArtistData, artistByID, artistAlias) = importData()

        // build the first model
        val bArtistAlias = sc.broadcast(artistAlias)

        val trainData = rawUserArtistData.map { line =>
            val Array(userID, artistID, count) = line.split(" ").map(_.toInt)
            val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
            Rating(userID, finalArtistID, count)
        }.cache()

        val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

        // see a feature vector
        // model.userFeatures.mapValues(_.mkString(", ")).first()

        spotCheckUser(userid=2093760, model=model, rawUserArtistData=rawUserArtistData, artistByID=artistByID)

    }
}
