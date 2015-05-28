import org.apache.spark.{SparkContext, SparkConf}
import StatsWM.statsWithMissing

case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)
case class Scored(md: MatchData, score: Double)

object RecordLinkage {
    val conf = new SparkConf().setMaster("local[*]").setAppName("RecordLinkage") //.set("spark.driver.memory", "12g")
    val sc = new SparkContext(conf)

    def main(args: Array[String]): Unit = {
        val dataFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/data/RecordLinkage/Blocks"
        val rawBlocks = sc.textFile(dataFolder)
        val head = rawBlocks.take(10)
        val line = head(5)

        def isHeader(line: String) = line.contains("id_1")
        val noHeader = rawBlocks.filter(!isHeader(_))

        def toDoubleNaN(s: String) = if (s == "?") Double.NaN else s.toDouble

        def parseLine(line: String): MatchData = {
            val pieces = line.split(",")
            val (id1, id2) = (pieces(0).toInt, pieces(1).toInt)
            val matched = pieces(11).toBoolean
            val rawScores = pieces.slice(2, 11)
            val scores = rawScores.map(toDoubleNaN)

            MatchData(id1, id2, scores, matched)
        }
        // val md = parseLine(line)

        val localMDS = head.filter(!isHeader(_)).map(parseLine)
        val parsed = noHeader.map(parseLine).cache()
        // data load done

        val localGrouped = localMDS.groupBy(_.matched)
        //localGrouped.mapValues(_.length).foreach(println)

        // histogram
        val matchCounts = parsed.map(_.matched).countByValue()
        //val matchCountsSeq = matchCounts.toSeq
        //matchCountsSeq.sortBy(_._2).reverse.foreach(println)

        // summary statistics for continuous variables
        //parsed.map(_.scores(0)).filter(!_.isNaN).stats()
        // use NAStatCounter
//        val nasRDD = parsed.map(_.scores.map(NAStatCounter(_)))
//        val reduced = nasRDD.reduce((n1, n2) => {
//            n1.zip(n2).map { case (a, b) => a merge b }
//        })

        // more efficient implementation of NAStats using NAStatCounter.statsWithMissing:
        val statsM = statsWithMissing(parsed.filter(_.matched).map(_.scores))
        val statsN = statsWithMissing(parsed.filter(!_.matched).map(_.scores))

        statsM.zip(statsN).zip(0 until 9).map { case ((m, n), fNo) =>
            ((m.missing + n.missing, m.stats.mean - n.stats.mean), fNo)
        }.foreach(println)

        // scoring model:
        def naZero(d: Double) = if(d.isNaN) 0.0 else d

        val ct = parsed.map(md => {
            val score = Array(2, 5, 6, 7 ,8).map(i => naZero(md.scores(i))).sum
            Scored(md, score)
        })

        println(ct.filter(s => s.score >= 4.0).map(s => s.md.matched).countByValue())
        println(ct.filter(s => s.score >= 2.0).map(s => s.md.matched).countByValue())
    }
}
