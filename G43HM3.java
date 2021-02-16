package bdc1819;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;


public class G43HM3 {

    /*
     * Need some elements on command line, instead use default values:
     * - file name (args[0])
     * - k (args[1])
     * - iter (args[2])
     */
    public static void main(String[] args){

        // default values
        String fileName = "covtype.data";
        int k = 50;
        int iter = 3;

        // read and check values in the command line
        if(args.length < 3)
            System.err.println("Not enought input values, use default");
        else{
            fileName = args[0];
            try{
                k = Integer.parseInt(args[1]);
                iter = Integer.parseInt(args[2]);
            } catch(NumberFormatException e) {
                e.printStackTrace();
            }
        }

        System.out.println("Begin computation with # of cluster: " + k + " and # of iteration: " + iter);

        long startTime = System.currentTimeMillis();

        // try to read the input file as an ArrayList<Vector>
        ArrayList<Vector> inputData = new ArrayList<>();
        try {
            inputData = readVectorsSeq(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // print an element
        //System.err.println(inputData.get(50));

        // try kmeans++
        // create an array with weight (all equal to 1)
        ArrayList<Long> WP = new ArrayList<>();
        for(int i=0; i<inputData.size(); i++)
            WP.add(1L);

        // kmeans++
        ArrayList<Vector> centers = kmeansPP(inputData, WP, k, iter);

        // test of object functions
        double obj = kmedianObj(inputData, centers);

        System.out.println("Obj value after Lloyd: " + obj);

        /*
        // print centers
        System.out.println("\nFound centers:");
        for(Vector c:centers)
            System.out.println(c);
        */

        long stopTime = System.currentTimeMillis();
        System.out.println("(Executed in: " + (stopTime - startTime) + "millis)");

    }

    /*
     * Kmeans++ and Lloyd
     *
     * @param P input set of points
     * @param WP weights for P
     * @param k number of points in output
     * @param iter max number of iterations
     *
     * @output C of k centers
     */

    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, int k, int iter){

        // check the correctness of input data
        if(P.size() != WP.size())
            return null;

        // useful variables and objects
        int elements = P.size(); // number of points in input
        ArrayList<Vector> S = new ArrayList<>(); // list of centers
        ArrayList<Double> Q = new ArrayList<>(); // list with distance from point (in P) to the closest center
        double Q_total = 0.; // total sum of Q vector, updated at every cycle

        //******************* k-means++ ....find C'
        Vector c1 = P.get((int)(Math.random()*(elements+1))); // select a random elements from P
        S.add(c1); // add it as first element of S

        // initialize distance from the only center up to now and the sum of it
        for(int i=0; i<P.size(); i++){ // for each elements
            Q.add(WP.get(i)* dist(c1, P.get(i))); //create and add the distance of i-th elements of P from the only one center
            Q_total += Q.get(i); // add to the sum of distances (to speed up the computation)
        }

        // q: element in P - S
        for(int i=2; i<k; i++){

            // w_p*(d_p)/(sum_{q non center} w_q*(d_q))

            // compute a random number x€[0,1]
            double choice = Math.random(); // random choice number
            //System.err.println("choice: " + choice + "\tQtotal: " + Q_total + "\tQ.get(i): " + Q.get(i));
            // search index that we need
            double cumProb = 0.; // the sum of the probability
            int l = 0; // the index of the chosen element
            while(cumProb < choice){ // until the sum of prob is lower thant the choice
                l++; //next index
                if(!S.contains(P.get(l))) //if S contains the l-th elements of P skip this
                    cumProb += (((double) WP.get(l) * Q.get(l)) / Q_total); // otherwise add the probability
            }
            //System.err.println("CumProb at iter: " + l + " is " + cumProb);
            // so the choice is l!

            // add element in S
            S.add(P.get(l));

            // recompute distance (if with the new center is lower than before) and sum over Q vector
            Q_total = 0; // reset to zero the total Q so it can be recomputed with new distances
            for(int j=0; j<P.size(); j++){ // for each elements in P (and so in Q)
                double dist = (double) WP.get(j) * dist(P.get(l), P.get(j)); // compute the distance
                if(dist < Q.get(j)){ // if distance is lower that before, update
                    Q.set(j, dist);
                }
                Q_total += Q.get(j); //update the total Q
                //System.err.println("Q_total: " + Q_total);
            }
        }

        System.out.println("Obj value before Lloyd: " + kmedianObj(P, S));

        //***************** Lloyd alg to find C
        double bestObj = Double.MAX_VALUE; //set to "infinity"

        for(int i=0; i<iter; i++){ // for each iterations
            ArrayList<ArrayList<Integer>> C_temp = partitionIndex(P, S); // create the partition with best center until now
            ArrayList<Vector> newS = new ArrayList<>(); // the c' vector, new possible centers [size: S.size()]
            for (int j=0; j<k-1; j++) // for every cluster
                newS.add(centroid(P, WP, C_temp.get(j))); //add its centroid to the new possible centers

            double newObj = kmedianObj(P, newS); // compute obj with new centers (newS)

            if(newObj < bestObj){ // if it's better that before update the stuff
                bestObj = newObj;
                S = newS;
                //System.err.println("Iter " + i + " with best obj= " + bestObj);
            }

        }

        return S;
    }


    /*
     * Partition method that return index. Weighted variant.
     * @param P points
     * @param S centers
     * @output list of Clusters (not the centers, use S!)
     */
    public static ArrayList<ArrayList<Integer>> partitionIndex(ArrayList<Vector> P, ArrayList<Vector> S){
        int k = S.size(); // number of centers (and so number of clusters)
        ArrayList<ArrayList<Integer>> C = new ArrayList<>(); // ArrayList of clusters (that are ArrayList of integer index)

        // initialize output list
        for(int i=0; i<k; i++){
            C.add(new ArrayList<>()); //create each cluster
            //C.get(i).add(S.get(i)); //C_i <-- {c_i} add center on each cluster
        }

        for(int i=0; i<P.size(); i++){ //cycle on each points (P)
            if(!S.contains(P.get(i))){ //check that we're on P - S, so skip elements that are both in S and P
                // argmin i=1...K of {d(p, c_i)}
                int l = 0; // the index of the elements with best distance up to now (initialize)
                double bestDist = dist(P.get(i), S.get(0)); // the best distance up to now (initialize)
                for(int j=1; j<k; j++){ // cycle on clusters
                    if(dist(P.get(i), S.get(j)) < bestDist){ //create new distance and check if it's the best
                        // new best distance
                        l = j; // update the index of it
                        bestDist = dist(P.get(i), S.get(j)); // and save the new best distance
                    }
                }

                // C_l <-- C_l U {p}
                if(!C.get(l).contains(i)) //if the l-cluster not already contain p
                    C.get(l).add(i); // add it

            }//close if
        }

        return C;
    }


    /*
     * Distance method
     *
     * @param a vector
     * @param b vector
     * @output double distance
     */
    public static double dist(Vector a, Vector b){
        if (a.size()!=b.size())
            return -1;
        return Math.sqrt(Vectors.sqdist(a,b));
    }


    /*
     * K-Means Objective Function
     *
     * @param P points
     * @param centers centers
     * @output K-Means objective function
     */

    public static double kmedianObj(ArrayList<Vector> P, ArrayList<Vector> centers){
        double obj = 0.;
        ArrayList<ArrayList<Integer>> part = partitionIndex(P, centers); // partition the data with the provided centers
        for(int i=0; i<part.size(); i++) { // for each cluster (and each relative center) (cycle on ArrayList<Vector>)
            for (int p:part.get(i)) // for each index of point belonging to this cluster (cycle on int)
                obj += dist(centers.get(i), P.get(p)); // add the distance to the objective function
        }
        return obj/P.size();
    }


    /*
     * Find centroid of a set of vector
     *
     * @param P points
     * @param WP weights
     * @param indexesOfPoints index of the points of the cluster
     * @output centroid point (as a Vector)
     */
    public static Vector centroid( ArrayList<Vector> P,  ArrayList<Long> WP, ArrayList<Integer> indexesOfPoints){ //TODO we need to pass also the weight
        // initialize the centroid and its size
        int N = 0; // total weight of points
        for(int i:indexesOfPoints)
            N+=WP.get(i);

        int D = P.get(0).size(); // number of dimensions of each vector
        double[] cent = new double[D]; // centroid with length equal to the dimensions of each vector (D)

        for(int index:indexesOfPoints){ // foreach (int) index of points (cycle on N elements)
            for(int i=0; i<D; i++) // foreach dimensions of our vectors (D)
                cent[i] += P.get(index).apply(i) * WP.get(index); // add at each dimensions values of the vector of the selected dimension
        }

        // division by N of every element in the centroid
        for (int i=0; i<D; i++)
            cent[i] /= N;

        return Vectors.dense(cent);
    }


    /*
     * Support method for readVectorsSeq
     */
    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    /*
     * Used for reading the points
     * @param filename name of the file as a string
     * @output ArrayList<Vector> of the read file
     */
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


}
