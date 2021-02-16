import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.DoubleRDDFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;

import static java.lang.Double.max;

public class G43HM1 {

    public static class StdComparator implements Serializable, Comparator<Double> {

        public int compare(Double a, Double b) {
            if (a < b) return -1;
            else if (a > b) return 1;
            return 0;
        }

    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        // Read a list of numbers from the program options
        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s =  new Scanner(new File(args[0]));
        while (s.hasNext()){
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);


        // compute and print max value with reduce function (a "manual" way)
        double max_manual = dNumbers.reduce((x, y) -> max(x,y));
        System.out.println("Max (computed in manual mode) is " + max_manual);

        // compute and print max value with max function (an "automatic" way)
        double max_automatic = dNumbers.max(new StdComparator());
        System.out.println("Max (computed in automatic mode) is " + max_automatic);

        // normalization
        JavaRDD<Double> dNumbersNormalized = dNumbers.map((x) -> x/max_automatic);

        // Compute the mean
        double mean = (dNumbersNormalized.reduce((x,y) -> x + y))/dNumbersNormalized.count(); //new function!
        System.out.println("The mean is " + mean);

        // normalization and first counts
        dNumbersNormalized = dNumbers.map((x) -> Math.pow(x/max_automatic - mean,2));

        // Compute the standard deviation
        double sd = Math.sqrt( dNumbersNormalized.reduce((x,y) -> x+y)/dNumbersNormalized.count() );
        System.out.println("The standard deviation is " + sd);


    }

}