package org.deeplearning4j.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * IMDB import
 *
 * @author Fariz Rahman
 */
public class ImdbKerasImport {

    private static final int maxlen = 256;
    private static Map<String, Integer> wordIndex;
    private static MultiLayerNetwork IMDBMODEL;

    private static void loadWordIndex() throws Exception {
        wordIndex = new HashMap<>();
        File file = new ClassPathResource("modelimport/keras/word_index.txt").getFile();
        String content = FileUtils.readFileToString(file);
        String[] lines = content.split("\n");
        for(int i=0; i < lines.length - 1; i++){
            String line = lines[i];
            String[] kv = line.split(",");
            String k = kv[0];
            int v = Integer.parseInt(kv[1]);
            wordIndex.put(k, v);

        }
    }

    private static INDArray encodeText(String text) throws Exception {
        String[] words = text.split(" ");
        double arr[] = new double[maxlen];
        int pads = 256 - words.length;
        for(int i = 0; i<pads; i++){
            arr[i] = (double)wordIndex.get("<PAD>");
        }
        for(int i=0; i<words.length; i++){
            if(wordIndex.containsKey(words[i]) ){
                arr[pads + i] = (double)wordIndex.get(words[i]);
            }
            else {
                arr[pads + i] = (double)wordIndex.get("<UNK>");
            }
        }
        INDArray indArr = Nd4j.create(arr).reshape(256);
        return indArr;
    }

    public static void loadModel(String filepath) throws Exception{
        String filePath = new ClassPathResource(filepath).getFile().getPath();
        IMDBMODEL = KerasModelImport.importKerasSequentialModelAndWeights(filePath);
    }

    public static double predict(INDArray arr){
        arr = Nd4j.expandDims(arr, 0);  // add batch dimension
        INDArray outArr = IMDBMODEL.output(arr);
        double pred = outArr.getDouble(0);
        return pred;
    }

    public static void main(String[] args) throws Exception{
        loadModel("modelimport/keras/imdb.h5");
        loadWordIndex();
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        while(true){
            System.out.println("Enter review : ");
            String review = reader.readLine();
            INDArray arr = encodeText(review);
            double prediction = predict(arr);
            System.out.println(String.format("Sentiment prediction : " + prediction));
        }

    }
}
