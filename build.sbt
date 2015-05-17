name := "spark-sandbox"

version := "0.0.1"

scalaVersion := "2.10.4"

// libraries
libraryDependencies ++= Seq(
    "org.apache.spark" % "spark-core_2.10" % "1.3.1" % "provided",
    "org.apache.spark" % "spark-mllib_2.10" % "1.3.1",
    "com.databricks" % "spark-csv_2.10" % "1.0.3"
    )
