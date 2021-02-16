package bdc1819;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.awt.*;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

/* Input from command line: file(s) to read and number of partition k. Something like:
 *      text-sample.txt 10000
 * Remember the VM options:
 *      -Dspark.master="local[*]"
 */

public class G43HM2{

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        // Parameters
        int k = Integer.parseInt(args[1]);

        // Spark configuration
        SparkConf conf =
                new SparkConf(true)
                        .setAppName("Word Count");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // Read from file
        JavaRDD<String> docs = sc.textFile(args[0]).cache();

        // Repartition
        docs = docs.repartition(k);
        docs.count();

        ////////////////////// Improved Word Count 1
        System.out.println("Word Count 1\n");

        // Profiling
        long start = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordcountpairs = docs

                // Map phase
                .flatMapToPair((document) -> {

                    // Create the k-v pairs
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();
                })

                // Reduce Phace
                .reduceByKey((x,y) -> x + y);

        System.out.println("Count after WC 1: " + wordcountpairs.count());
        long end = System.currentTimeMillis();
        System.out.println("Elapsed time WC 1: " + (end - start) + " ms\n");



        //////////////// Improved Word Count 2
        System.out.println("Word Count 2\n");
        start = System.currentTimeMillis();

        // ROUND 1
        Random rnd = new Random();
        JavaPairRDD<String, Long> wordcount2pairs = docs

                // Map phase
                .flatMapToPair((document) -> {

                    // Create the k-v pairs
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();
                })

                .groupBy((s)-> (long)(rnd.nextInt(k)))

                // Reduce phase (x,(w,c_i(w))->(w, c(x, w))
                .flatMapToPair((document) -> {
                    HashMap<String,Long> hash = new HashMap<>();

                    // Count occurences
                    for (Tuple2<String,Long> c : document._2)
                    {
                        if (hash.get(c._1)==null)
                        {
                            hash.put(c._1,c._2);
                        }
                        else {
                            long val = hash.get(c._1);
                            hash.put(c._1, c._2 + val);
                        }
                    }

                    // Prepare data to be returned
                    ArrayList<Tuple2<String,Long>> fin = new ArrayList<>();
                    Iterator it = hash.entrySet().iterator();
                    while (it.hasNext())
                    {
                        Map.Entry m = (Map.Entry)it.next();
                        fin.add(new Tuple2<>((String)m.getKey(),(Long)m.getValue()));
                    }
                    return fin.iterator();

                })

                // ROUND2
                // Map Phase -> IDENTITY FUNCTION
                // Reduce Phase
                .reduceByKey((x,y) -> x + y);

        System.out.println("Count after WC 2: " + wordcount2pairs.count());
        end = System.currentTimeMillis();
        System.out.println("Elapsed time WC 2: " + (end - start) + " ms\n");

        //////////////// Improved Word Count 2.1
        System.out.println("Word Count 2.1\n");

        start = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordcount3pairs = docs

                // ROUND 1
                // Map phase
                .flatMapToPair((document) -> {

                    // Create the k-v pairs
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();
                })

                .mapPartitionsToPair((document) -> {

                    HashMap<String,Long> hash = new HashMap<>();

                    // Count occurences
                    while(document.hasNext())
                    {
                        Tuple2<String, Long> c = document.next();
                        if (hash.get(c._1)==null)
                        {
                            hash.put(c._1, c._2);
                        }
                        else {
                            long val = hash.get(c._1);
                            hash.put(c._1, c._2 + val);
                        }
                    }

                    // Prepare data to be returned
                    ArrayList<Tuple2<String,Long>> fin = new ArrayList<>();
                    Iterator it = hash.entrySet().iterator();
                    while (it.hasNext())
                    {
                        Map.Entry m = (Map.Entry)it.next();
                        fin.add(new Tuple2<>((String)m.getKey(),(Long)m.getValue()));
                    }
                    return fin.iterator();
                })

                // ROUND 2
                // Map Phase -> IDENTITY FUNCTION
                //Reduce Phase
                .reduceByKey((x,y) -> x + y);

        System.out.println("Count after WC 2.1: " + wordcount3pairs.count());
        end = System.currentTimeMillis();
        System.out.println("Elapsed time WC 2.1: " + (end - start) + " ms\n");

        //////////////// Average Length
        JavaRDD<Integer> lengths = wordcount3pairs.map((word) -> word._1().length());

        double mean = (double)lengths.reduce((x,y) -> x + y)/(double)wordcount3pairs.count();

        System.out.println("Average length of the distinct words is: " + mean + "\n");

        // Wait!
        System.out.println("Press enter to finish");
        try {
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
