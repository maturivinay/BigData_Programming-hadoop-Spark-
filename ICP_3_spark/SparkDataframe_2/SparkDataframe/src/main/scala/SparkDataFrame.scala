import org.apache.spark.sql.SparkSession

object SparkDataFrame {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "C:\\downloads\\Winutils")
    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val df = spark.read.format("csv").option("header", "true").load("C:\\Users\\matur\\Desktop\\UMKC\\bigdata_programming\\ICP_3_spark\\SparkDataframe_2\\SparkDataframe\\src\\main\\scala\\ConsumerComplaints.csv")

    df.show()


    //df.write.format("csv").save("C:\\Users\\matur\\Desktop\\UMKC\\123")


    df.groupBy("Zip Code").count().show()

    df.registerTempTable("Consumer")

    val ct = spark.sql("select count(*) from Consumer where Company = 'Citibank' ")
    print("Citibank employee count :")
    ct.show()

    val csct = spark.sql("select * from Consumer where Company = 'Citibank' ")
    print("Citibank")
    csct.show()

    val cscw = spark.sql("select * from Consumer where Company = 'Wells Fargo & Company' ")
    print("Wells Fargo & Company")
    cscw.show()

    val unionDf = csct.union(cscw)
    unionDf.show()





    val df3 = df.limit(50)
    val df4 = df.limit(80)

    df3.createOrReplaceTempView("left")
    df4.createOrReplaceTempView("right")


    val join_aggregate = spark.sql("select left.ProductName,right.Company FROM left,right where left.ComplaintID = " +
      "right.ComplaintID")
    join_aggregate.show()



    val th = df.take(13).last
    print("13th row")
    print(th)
  }
}



