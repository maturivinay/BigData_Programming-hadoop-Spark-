import org.apache.spark.sql.SparkSession

object Friends {
  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "C:\\winutils")

    val sc = SparkSession
      .builder
      .appName("SparkWordCount")
      .master("local[*]")
      .getOrCreate().sparkContext

    val flist = sc.textFile("C:\\Users\\matur\\Desktop\\UMKC\\bigdata_programming\\Lab_Assignment_3\\src\\main\\scala\\abc.txt")
    val common = flist.flatMap { x =>
      val splitflist = x.split(" : ")
      val owner = splitflist(0)
      val friendslist = splitflist(1).split(" ")
      friendslist.foreach(println)
      val makelist = friendslist.slice(0, friendslist.size).map(y => {
        if (owner > y) (y, owner) else (owner, y)
      })
      makelist.map(z => (z, friendslist.slice(0, friendslist.size).toSet))
    }
    val findcommon = common.reduceByKey((x, y) => x intersect y).sortByKey()
    findcommon.collect().take(10).foreach(x => {
      println(s"${x._1} -> (${x._2.mkString(" ")})")
      findcommon.saveAsTextFile("C:\\Users\\matur\\Desktop\\UMKC\\bigdata_programming\\Lab_Assignment_3\\src\\main\\scala\\output_facebook")
    })


  }
}