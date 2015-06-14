/**
 * Created by sukruhasdemir on 13/06/15.
 */

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.util.MLUtils.kFold

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Recommender {

    val conf = new SparkConf().setMaster("local[*]").setAppName("Recommender").set("spark.driver.memory", "8g")
    val sc = new SparkContext(conf)

    val dataLocation = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/Recommenders/profiledata_06-May-2005/"
    val userArtistDataFile = dataLocation + "user_artist_data.txt"
    val artistDataFile = dataLocation + "artist_data.txt"
    val artistAliasFile = dataLocation + "artist_alias.txt"

    def importData(): (RDD[String], RDD[(Int, String)], Map[Int, Int]) = {  // whatever
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
        }.collectAsMap().toArray.toMap  // to get immutable map

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

        val recommendations = model.recommendProducts(userid, 5)
        recommendations foreach println

        val recommendedProductIDs = recommendations.map(_.product).toSet

        artistByID.filter { case (id, name) => recommendedProductIDs.contains(id) }.values.collect().foreach(println)
    }

    def areaUnderCurve(positiveData: RDD[Rating], bAllItemIDs: Broadcast[Array[Int]],
                       predictFunction: (RDD[(Int,Int)] => RDD[Rating])) = {
        // What this actually computes is AUC, per user. The result is actually something
        // that might be called "mean AUC".

        // Take held-out data as the "positive", and map to tuples
        val positiveUserProducts = positiveData.map(r => (r.user, r.product))
        // Make predictions for each of them, including a numeric score, and gather by user
        val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

        // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
        // small AUC problems, and it would be inefficient, when a direct computation is available.

        // Create a set of "negative" products for each user. These are randomly chosen
        // from among all of the other items, excluding those that are "positive" for the user.
        val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
            // mapPartitions operates on many (user,positive-items) pairs at once
            userIDAndPosItemIDs => {
                // Init an RNG and the item IDs set once for partition
                val random = new Random()
                val allItemIDs = bAllItemIDs.value
                userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
                    val posItemIDSet = posItemIDs.toSet
                    val negative = new ArrayBuffer[Int]()
                    var i = 0
                    // Keep about as many negative examples per user as positive.
                    // Duplicates are OK
                    while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
                        val itemID = allItemIDs(random.nextInt(allItemIDs.size))
                        if (!posItemIDSet.contains(itemID)) {
                            negative += itemID
                        }
                        i += 1
                    }
                    // Result is a collection of (user,negative-item) tuples
                    negative.map(itemID => (userID, itemID))
                }
            }
        }.flatMap(t => t)
        // flatMap breaks the collections above down into one big set of tuples

        // Make predictions on the rest:
        val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

        // Join positive and negative by user
        positivePredictions.join(negativePredictions).values.map {
            case (positiveRatings, negativeRatings) =>
                // AUC may be viewed as the probability that a random positive item scores
                // higher than a random negative one. Here the proportion of all positive-negative
                // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
                var correct = 0L
                var total = 0L
                // For each pairing,
                for (positive <- positiveRatings;
                     negative <- negativeRatings) {
                    // Count the correctly-ranked pairs
                    if (positive.rating > negative.rating) {
                        correct += 1
                    }
                    total += 1
                }
                // Return AUC: fraction of pairs ranked correctly
                correct.toDouble / total
        }.mean() // Return mean AUC over users
    }

    def buildRatings(rawUserArtistData: RDD[String], bArtistAlias: Broadcast[Map[Int,Int]]) = {
        rawUserArtistData.map { line =>
            val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
            val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
            Rating(userID, finalArtistID, count)
        }
    }

    def predictMostListened(sc: SparkContext, train: RDD[Rating])(allData: RDD[(Int, Int)]) = {
        val bListenCount = sc.broadcast(
            train.map(r => (r.product, r.rating))
                 .reduceByKey(_ + _).collectAsMap()
        )
        allData.map { case (user, product) =>
            Rating(
                user,
                product,
                bListenCount.value.getOrElse(product, 0.0)
            )
        }
    }

    def crossValidation(allData: RDD[Rating], bAllItemIDs: Broadcast[Array[Int]], numFolds: Int, seed: Int ): Double = {

        val cvDataSets = kFold(rdd=allData, numFolds=numFolds, seed=seed)

        // cache data sets
        for (tup <- cvDataSets) {
            tup._1.cache()
            tup._2.cache()
        }

        val cvAUCs = cvDataSets.map { case (cvTrainingData, cvValidationData) =>
            val cvModel = ALS.trainImplicit(cvTrainingData, 10, 5, 0.01, 1.0)
            areaUnderCurve(cvValidationData, bAllItemIDs, cvModel.predict)
        }

        cvAUCs.sum / numFolds
    }

    def main(Args: Array[String]): Unit = {
        val (rawUserArtistData, artistByID, artistAlias) = importData()

        // build the first model
        val bArtistAlias = sc.broadcast(artistAlias)

        val fullTrainData = buildRatings(rawUserArtistData, bArtistAlias).cache()

        val fullModel = ALS.trainImplicit(fullTrainData, 10, 5, 0.01, 1.0)

        // see a feature vector
        // model.userFeatures.mapValues(_.mkString(", ")).first()

        spotCheckUser(userid=2093760, model=fullModel, rawUserArtistData=rawUserArtistData, artistByID=artistByID)

        // compute AUC
        val allData = buildRatings(rawUserArtistData, bArtistAlias)
        val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
        trainData.cache()
        cvData.cache()

        val allItemIDs = allData.map(_.product).distinct().collect()  // remove duplicates, collect to driver
        val bAllItemIDs = sc.broadcast(allItemIDs)

        val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
        val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict) // 0.9659313568834662
        println("AUC is: " + auc)

        // try with 10-fold CV
        val CVAUC = crossValidation(allData = allData, bAllItemIDs=bAllItemIDs, numFolds=10, seed=500)
        println("10-fold CV meanAUCs: " + CVAUC)  // 0.9657541401455754

        // how does the dummy model do?
        val mostListenedAUC = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))
        println("Dummy model AUC: " + mostListenedAUC)  // 0.9399350367758861

        // grid search for hyperparameters
        val evaluations =
            for {rank <- Array(10, 50)
                 lambda <- Array(1.0, 0.0001)
                 alpha <- Array(1.0, 40.0)}
            yield {
                val gsModel = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
                val gsAuc = areaUnderCurve(cvData, bAllItemIDs, gsModel.predict)
                ((rank, lambda, alpha), gsAuc)
            }
        println("Grid search for hyperparameters: ")
        evaluations.sortBy(_._2).reverse.foreach(println)
    }
}
