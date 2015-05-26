import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.sql.{DataFrame, SQLContext, Column}
import org.apache.spark.sql.functions._
import com.databricks.spark.csv._


object FilePaths {
    val mainFolder = "/Users/sukruhasdemir/Repos/Courses/spark-sandbox/"
    val dataFolder = mainFolder + "data/titanic/"
    val trainDataFile = dataFolder + "train.csv"
    val testDataFile = dataFolder + "test.csv"
    val resultsFolder = mainFolder + "results/"
}


object TitanicUDFs {
    val toInt = udf[Int, String](_.toInt)
    val toDouble = udf[Double, String](_.toDouble)
    val classUDF = udf[Double, String](_.toDouble - 1.0)  // categorical variables start from 0 in MLLib
    val genderUDF = udf[Double, String](gen => if (gen == "male") 1.0 else 0.0)
    val ageUDF = udf[Option[Double], String](rawAge => if (rawAge == "") None else Some(rawAge.toDouble))
    val fareUDF = udf[Option[Double], String](rawPrice => if (rawPrice == "") None else Some(rawPrice.toDouble))
}


case class TitanicResult(PassengerId: Int, Survived: Int)
case class ClassGenderAge(Class: Double, Gender: Double, AgeMash: Option[Double])
case class ClassPrice(Class: Double, Price: Option[Double])


object Titanic {
    val conf = new SparkConf().setMaster("local[*]").setAppName("Titanic")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    type SparkVector = org.apache.spark.mllib.linalg.Vector
    type fullDataTuple = (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], RDD[(Int, SparkVector)])

    import TitanicUDFs._
    import FilePaths._
    val trainingCSV = sqlContext.csvFile(trainDataFile, useHeader = true).cache()
    val testCSV = sqlContext.csvFile(testDataFile, useHeader = true).cache()


    def prepGenderClassData(): fullDataTuple = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .select("Id", "Survival", "Class", "Gender")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .select("Id", "Class", "Gender")

        val trainingFeatures = trainingDataCasted.map(row =>
            LabeledPoint(row.getInt(1), Vectors.dense(row.getDouble(2), row.getDouble(3)))).cache() // label, features
        val testFeatures = testDataCasted.map(row => (row.getInt(0), Vectors.dense(row.getDouble(1), row.getDouble(2)))) // id, features

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassSibSpData(): fullDataTuple = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("SibSpouse", toDouble(trainingCSV("SibSp")))
                .select("Id", "Survival", "Class", "Gender", "SibSpouse")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("SibSpouse", toDouble(testCSV("SibSp")))
                .select("Id", "Class", "Gender", "SibSpouse")

        val trainingFeatures = trainingDataCasted.map(row =>
            LabeledPoint(row.getInt(1), Vectors.dense(row.getDouble(2), row.getDouble(3), row.getDouble(4)))).cache()

        val testFeatures = testDataCasted.map(row =>
            (row.getInt(0), Vectors.dense(row.getDouble(1), row.getDouble(2), row.getDouble(3))))

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassSibSpParchFareData(): fullDataTuple = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("SibSpouse", toDouble(trainingCSV("SibSp")))
                .withColumn("ParCh", toDouble(trainingCSV("Parch")))
                .withColumn("Price", fareUDF(trainingCSV("Fare")))
                .select("Id", "Survival", "Class", "Gender", "SibSpouse", "ParCh", "Price")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("SibSpouse", toDouble(testCSV("SibSp")))
                .withColumn("ParCh", toDouble(testCSV("Parch")))
                .withColumn("Price", fareUDF(testCSV("Fare")))
                .select("Id", "Class", "Gender", "SibSpouse", "ParCh", "Price")

        // Price can be missing. Fill in average price by class instead.
        val allPriceData = trainingDataCasted.select("Class", "Price").rdd.union(
            testDataCasted.select("Class", "Price").rdd)

        val allPriceDF = allPriceData.map(row =>
            new ClassPrice(Class=row.getDouble(0), Price=if (row.isNullAt(1)) None else Some(row.getDouble(1)))
        ).toDF()

        val allPriceAverages = allPriceDF.groupBy("Class").avg("Price").collect()
            .map(row => (row.getDouble(0), row.getDouble(1))).toMap

        val trainingFeatures = trainingDataCasted.map { row =>
            val pSurvival = row.getInt(1)
            val pClass = row.getDouble(2)
            val pGender = row.getDouble(3)
            val pSibSpouse = row.getDouble(4)
            val pParCh = row.getDouble(5)
            val pPrice = if (row.isNullAt(6)) allPriceAverages(pClass) else row.getDouble(6)

            LabeledPoint(pSurvival, Vectors.dense(pClass, pGender, pSibSpouse, pParCh, pPrice))
        }.cache()

        val testFeatures = testDataCasted.map { row =>
            val pId = row.getInt(0)
            val pClass = row.getDouble(1)
            val pGender = row.getDouble(2)
            val pSibSpouse = row.getDouble(3)
            val pParCh = row.getDouble(4)
            val pPrice = if (row.isNullAt(5)) allPriceAverages(pClass) else row.getDouble(5)

            (pId, Vectors.dense(pClass, pGender, pSibSpouse, pParCh, pPrice))
        }

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassFamilyData(): fullDataTuple = {
        val _trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("SibSpouse", toDouble(trainingCSV("SibSp")))
                .withColumn("ParCh", toDouble(trainingCSV("Parch")))

        val trainingDataCasted = _trainingDataCasted
                .withColumn("FamilySize", _trainingDataCasted("SibSpouse") + _trainingDataCasted("ParCh"))
                .select("Id", "Survival", "Class", "Gender", "FamilySize")

        val _testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("SibSpouse", toDouble(testCSV("SibSp")))
                .withColumn("ParCh", toDouble(testCSV("Parch")))

        val testDataCasted = _testDataCasted
                .withColumn("FamilySize", _testDataCasted("SibSpouse") + _testDataCasted("ParCh"))
                .select("Id", "Class", "Gender", "FamilySize")

        val trainingFeatures = trainingDataCasted.map { row =>
            val pSurvival = row.getInt(1)
            val pClass = row.getDouble(2)
            val pGender = row.getDouble(3)
            val pFamilySize = row.getDouble(4)

            LabeledPoint(pSurvival, Vectors.dense(pClass, pGender, pFamilySize))
        }.cache()

        val testFeatures = testDataCasted.map { row =>
            val pId = row.getInt(0)
            val pClass = row.getDouble(1)
            val pGender = row.getDouble(2)
            val pFamilySize = row.getDouble(3)

            (pId, Vectors.dense(pClass, pGender, pFamilySize))
        }

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassSibSpFareData(): fullDataTuple = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("SibSpouse", toDouble(trainingCSV("SibSp")))
                .withColumn("Price", fareUDF(trainingCSV("Fare")))
                .select("Id", "Survival", "Class", "Gender", "SibSpouse", "Price")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("SibSpouse", toDouble(testCSV("SibSp")))
                .withColumn("Price", fareUDF(testCSV("Fare")))
                .select("Id", "Class", "Gender", "SibSpouse", "Price")

        // Price can be missing. Fill in average price by class instead.
        val allPriceData = trainingDataCasted.select("Class", "Price").rdd.union(
            testDataCasted.select("Class", "Price").rdd)

        val allPriceDF = allPriceData.map(row =>
            new ClassPrice(Class=row.getDouble(0), Price=if (row.isNullAt(1)) None else Some(row.getDouble(1)))
        ).toDF()

        val allPriceAverages = allPriceDF.groupBy("Class").avg("Price").collect()
                .map(row => (row.getDouble(0), row.getDouble(1))).toMap

        val trainingFeatures = trainingDataCasted.map { row =>
            val pSurvival = row.getInt(1)
            val pClass = row.getDouble(2)
            val pGender = row.getDouble(3)
            val pSibSpouse = row.getDouble(4)
            val pPrice = if (row.isNullAt(5)) allPriceAverages(pClass) else row.getDouble(5)

            LabeledPoint(pSurvival, Vectors.dense(pClass, pGender, pSibSpouse, pPrice))
        }.cache()

        val testFeatures = testDataCasted.map { row =>
            val pId = row.getInt(0)
            val pClass = row.getDouble(1)
            val pGender = row.getDouble(2)
            val pSibSpouse = row.getDouble(3)
            val pPrice = if (row.isNullAt(4)) allPriceAverages(pClass) else row.getDouble(4)

            (pId, Vectors.dense(pClass, pGender, pSibSpouse, pPrice))
        }

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassFareData(): fullDataTuple = {
        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("Price", fareUDF(trainingCSV("Fare")))
                .select("Id", "Survival", "Class", "Gender", "Price")

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("Price", fareUDF(testCSV("Fare")))
                .select("Id", "Class", "Gender", "Price")

        // Price can be missing. Fill in average price by class instead.
        val allPriceData = trainingDataCasted.select("Class", "Price").rdd.union(
            testDataCasted.select("Class", "Price").rdd)

        val allPriceDF = allPriceData.map(row =>
            new ClassPrice(Class=row.getDouble(0), Price=if (row.isNullAt(1)) None else Some(row.getDouble(1)))
        ).toDF()

        val allPriceAverages = allPriceDF.groupBy("Class").avg("Price").collect()
                .map(row => (row.getDouble(0), row.getDouble(1))).toMap

        val trainingFeatures = trainingDataCasted.map { row =>
            val pSurvival = row.getInt(1)
            val pClass = row.getDouble(2)
            val pGender = row.getDouble(3)
            val pPrice = if (row.isNullAt(4)) allPriceAverages(pClass) else row.getDouble(4)

            LabeledPoint(pSurvival, Vectors.dense(pClass, pGender, pPrice))
        }.cache()

        val testFeatures = testDataCasted.map { row =>
            val pId = row.getInt(0)
            val pClass = row.getDouble(1)
            val pGender = row.getDouble(2)
            val pPrice = if (row.isNullAt(3)) allPriceAverages(pClass) else row.getDouble(3)

            (pId, Vectors.dense(pClass, pGender, pPrice))
        }

        val splits = trainingFeatures.randomSplit(Array(0.7, 0.3), seed=12345L)
        val (initialTrainingFeatures, validationFeatures) = (splits(0).cache(), splits(1).cache())

        (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures)
    }


    def prepGenderClassAgeData(): (RDD[LabeledPoint], RDD[(Int, SparkVector)]) = {
        // lets not do validation or real testing this time. will just use age averages from all data, train on all
        // training and testing data, submit results for testing data

        // so both trainingFeatures and testFeatures will have imputed AgeMash variables as average of class and sex

        val trainingDataCasted = trainingCSV
                .withColumn("Id", toInt(trainingCSV("PassengerId")))
                .withColumn("Survival", toInt(trainingCSV("Survived")))
                .withColumn("Class", classUDF(trainingCSV("Pclass")))
                .withColumn("Gender", genderUDF(trainingCSV("Sex")))
                .withColumn("AgeMash", ageUDF(trainingCSV("Age")))
                .select("Id", "Survival", "Class", "Gender", "AgeMash").cache()

        val testDataCasted = testCSV
                .withColumn("Id", toInt(testCSV("PassengerId")))
                .withColumn("Class", classUDF(testCSV("Pclass")))
                .withColumn("Gender", genderUDF(testCSV("Sex")))
                .withColumn("AgeMash", ageUDF(testCSV("Age")))
                .select("Id", "Class", "Gender", "AgeMash").cache()

        // DataFrame.unionAll() ignores columns with missing values, so I will calculate the combined age averages
        // by converting to RDD, merge, then make DF again
        val allAgeDataRDD = trainingDataCasted.select("Class", "Gender", "AgeMash").rdd.union(
            testDataCasted.select("Class", "Gender", "AgeMash").rdd)
        // toDF implicit conversion doesn't seem to be working for an RDD of Row objects. needed to create an RDD of a
        // case class:
        val allAgeDataDF = allAgeDataRDD.map(row =>
            new ClassGenderAge(Class=row.getDouble(0),
                               Gender=row.getDouble(1),
                               AgeMash=if (row.isNullAt(2)) None else Some(row.getDouble(2)))).toDF()

        val allAgeAverages = allAgeDataDF.groupBy("Class", "Gender").avg("AgeMash").collect()
                .map(row => (row.getDouble(0), row.getDouble(1)) -> row.getDouble(2)).toMap

        val trainingFeatures = trainingDataCasted.map { row =>
            val pSurvived = row.getInt(1)
            val pClass = row.getDouble(2)
            val pGender = row.getDouble(3)
            val pAge = if (row.isNullAt(4)) allAgeAverages((pClass, pGender)) else row.getDouble(4)

            LabeledPoint(pSurvived, Vectors.dense(pClass, pGender, pAge))
        }.cache()

        val testFeatures = testDataCasted.map { row =>
            val pId = row.getInt(0)
            val pClass = row.getDouble(1)
            val pGender = row.getDouble(2)
            val pAge = if (row.isNullAt(3)) allAgeAverages((pClass, pGender)) else row.getDouble(3)

            (pId, Vectors.dense(pClass, pGender, pAge))
        }.cache()

        (trainingFeatures, testFeatures)
    }


    def runLRModels(dataInputFunction: () => fullDataTuple, outputFolderName: String): Unit = {
        val (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures) = dataInputFunction()

        val initialLRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(initialTrainingFeatures)
        println("*********")
        println("initial" + outputFolderName + " weights: " + initialLRModel.weights + " " + initialLRModel.intercept)

        val LRValidationResults = validationFeatures.map {
            case LabeledPoint(label, features) => (initialLRModel.predict(features), label)
        }.cache()

        val classificationError = LRValidationResults.filter(tuple => tuple._1 != tuple._2).count().toDouble /
                LRValidationResults.count()

        println("initial" + outputFolderName + " classification error rate: " + classificationError)
        val validationMetrics = new MulticlassMetrics(LRValidationResults)
        println("initial" + outputFolderName + " precision: " + validationMetrics.precision)
        println("initial" + outputFolderName + " recall: " + validationMetrics.recall)

        // train full model
        val LRFullModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(trainingFeatures)

        // evaluate over test data
        val LRFullResults = testFeatures.map {
            case (idInt, fVector) => (idInt, LRFullModel.predict(fVector).toInt)
        }.cache()

        println(outputFolderName + " weights: " + LRFullModel.weights + " " + LRFullModel.intercept)
        val LRResultsDF = LRFullResults.map(tuple => new TitanicResult(tuple._1, tuple._2)).toDF()

        LRResultsDF.show()
        LRResultsDF.saveAsCsvFile(resultsFolder + outputFolderName)
    }


    def runSeparateGenderModels(dataInputFunction: () => fullDataTuple, outputFolderName: String): Unit = {
        val (trainingFeatures, initialTrainingFeatures, validationFeatures, testFeatures) = dataInputFunction()

        // assuming Gender is at position 1 at the training and testing data features
        val initialMaleFeatures = initialTrainingFeatures.filter(point => point.features(3) == 1)
        val initialFemaleFeatures = initialTrainingFeatures.filter(point => point.features(3) == 0)

        val trainingMaleFeatures = trainingFeatures.filter(point => point.features(3) == 1)
        val trainingFemaleFeatures = trainingFeatures.filter(point => point.features(3) == 0)

        class DoubleLRModels(maleModel: LogisticRegressionModel,femaleModel: LogisticRegressionModel) {
            def predict(dataPoint: org.apache.spark.mllib.linalg.Vector): Double = {
                val dataPointGender = dataPoint(2)

                if (dataPointGender == 0)
                    femaleModel.predict(dataPoint)
                else
                    maleModel.predict(dataPoint)
            }

            def weights = "Male model weights: " + maleModel.weights + "\nFemale model weights: " + femaleModel.weights
            def intercept = "Male model intercept: " + maleModel.intercept + "\nFemale model intercept: "
                + femaleModel.intercept
        }

        val initialMaleModel =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(initialMaleFeatures)
        val initialFemaleModel =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(initialFemaleFeatures)

        val initialDoubleModels = new DoubleLRModels(initialMaleModel, initialFemaleModel)

        val LRValidationResults = validationFeatures.map {
            case LabeledPoint(label, features) => (initialDoubleModels.predict(features), label)
        }.cache()

        val classificationError = LRValidationResults.filter(tuple => tuple._1 != tuple._2).count().toDouble /
                LRValidationResults.count()

        println("initial" + outputFolderName + " classification error rate: " + classificationError)
        val validationMetrics = new MulticlassMetrics(LRValidationResults)
        println("initial" + outputFolderName + " precision: " + validationMetrics.precision)
        println("initial" + outputFolderName + " recall: " + validationMetrics.recall)

        // train the model on all training data
        val fulllMaleModel =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(trainingMaleFeatures)
        val fullFemaleModel =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
                .run(trainingFemaleFeatures)

        val fullDoubleModels = new DoubleLRModels(fulllMaleModel, fullFemaleModel)

        // evaluate over test data
        val LRFullResults = testFeatures.map {
            case (idInt, fVector) => (idInt, fullDoubleModels.predict(fVector).toInt)
        }.cache()

        println(outputFolderName + " weights: " + fullDoubleModels.weights + " " + fullDoubleModels.intercept)
        val LRResultsDF = LRFullResults.map(tuple => new TitanicResult(tuple._1, tuple._2)).toDF()

        LRResultsDF.show()
        LRResultsDF.saveAsCsvFile(resultsFolder + outputFolderName)

    }


    def runGenderClassAgeLRModel(): Unit = {
        val (trainingFeatures, testFeatures) = prepGenderClassAgeData()

        val LRModel = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).setValidateData(true)
            .run(trainingFeatures)

        println("*********")
        println("GenderClassAgeLRModel weights: " + LRModel.weights + " " + LRModel.intercept)

        val testResults = testFeatures.map {
            case (idInt, fVector) => (idInt, LRModel.predict(fVector).toInt)
        }.cache()

        val testResultsDF = testResults.map(tuple => new TitanicResult(tuple._1, tuple._2)).toDF()
        testResultsDF.saveAsCsvFile(resultsFolder + "LRGenderClassAgeResults")
    }


    def main(args: Array[String]): Unit = {
        runGenderClassAgeLRModel()

        runLRModels(prepGenderClassData, "LRGenderClassResults")
        runLRModels(prepGenderClassSibSpData, "LRGenderClassSibSpResults")
        runLRModels(prepGenderClassSibSpParchFareData, "LRGenderClassSibSpParchFareResults")
        runLRModels(prepGenderClassFareData, "LRGenderClassFareResults")
        runLRModels(prepGenderClassSibSpFareData, "LRGenderClassSibSpFareResults")
        runLRModels(prepGenderClassFamilyData, "LRGenderClassFamilyModel")

        runSeparateGenderModels(prepGenderClassFamilyData, "LRGenderClassFamilyModelSEP")
    }
}
