import org.apache.spark.util.StatCounter

class NAStatCounter extends Serializable {
    val sCounter = new StatCounter()
    var missing = 0L

    def add(x: Double): NAStatCounter = {
        if (x.isNaN)
            missing += 1
        else
            sCounter.merge(x)

        this
    }

    def merge(other: NAStatCounter): NAStatCounter = {
        sCounter.merge(other.sCounter)
        missing += other.missing
        this
    }

    override def toString = "stats: " + sCounter.toString + " NaN " + missing
}

object NAStatCounter extends Serializable {
    def apply(x: Double) = new NAStatCounter().add(x)
}
