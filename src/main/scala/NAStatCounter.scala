import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter

class NAStatCounter extends Serializable {
    val stats = new StatCounter()
    var missing = 0L

    def add(x: Double): NAStatCounter = {
        if (x.isNaN)
            missing += 1
        else
            stats.merge(x)

        this
    }

    def merge(other: NAStatCounter): NAStatCounter = {
        stats.merge(other.stats)
        missing += other.missing
        this
    }

    override def toString = "stats: " + stats.toString + " NaN: " + missing
}

object NAStatCounter extends Serializable {
    def apply(x: Double) = new NAStatCounter().add(x)

    def statsWithMissing(dataRDD: RDD[Array[Double]]): Array[NAStatCounter] = {
        val naStats = dataRDD.mapPartitions((iter: Iterator[Array[Double]]) => {
            val nas: Array[NAStatCounter] = iter.next().map(NAStatCounter(_))
            iter.foreach(arr => {
                nas.zip(arr).foreach { case (n, d) => n add d }
            })
            Iterator(nas)
        })

        naStats.reduce((n1, n2) => {
            n1.zip(n2).map { case (a, b) => a merge b }
        })
    }
}
