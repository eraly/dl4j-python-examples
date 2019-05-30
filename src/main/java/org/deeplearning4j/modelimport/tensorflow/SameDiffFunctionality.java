package org.deeplearning4j.modelimport.tensorflow;

import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author susaneraly
 */
public class SameDiffFunctionality {
    //Python code for this can be found in resources/import/tensorflow under generate_model.py and freeze_model_after.py
    //Input node/Placeholder in this graph is names "input"
    //Output node/op in this graph is names "output"
    public final static String BASE_DIR = "modelimport/tensorflow";

    public static void main(String[] args) throws Exception {
        final String FROZEN_MLP = new ClassPathResource(BASE_DIR + "/frozen_model.pb").getFile().getPath();

        //Load placeholder inputs and corresponding predictions generated from tensorflow
        Map<String, INDArray> inputsPredictions = readPlaceholdersAndPredictions();

        //Load the graph into samediff
        SameDiff graph = TFGraphMapper.getInstance().importGraph(new File(FROZEN_MLP));
        //libnd4j executor
        //running with input_a array expecting to get prediction_a
        graph.associateArrayWithVariable(inputsPredictions.get("input_a"), graph.variableMap().get("input"));
        NativeGraphExecutioner executioner = new NativeGraphExecutioner();
        INDArray[] results = executioner.executeGraph(graph); //returns an array of the outputs
        INDArray libnd4jPred = results[0];
        System.out.println("LIBND4J exec prediction for input_a:\n" + libnd4jPred);
        if (libnd4jPred.equals(inputsPredictions.get("prediction_a"))) {
            //this is true and therefore predictions are equal
            System.out.println("Predictions are equal to tensorflow");
        } else {
            throw new RuntimeException("Predictions don't match!");
        }

        //Now to run with the samediff executor, with input_b array expecting to get prediction_b
        SameDiff graphSD = TFGraphMapper.getInstance().importGraph(new File(FROZEN_MLP)); //Reimport graph here, necessary for the 1.0 alpha release
        graphSD.associateArrayWithVariable(inputsPredictions.get("input_b"), graph.variableMap().get("input"));
        INDArray samediffPred = graphSD.execAndEndResult();
        System.out.println("SameDiff exec prediction for input_b:\n" + samediffPred);
        if (samediffPred.equals(inputsPredictions.get("prediction_b"))) {
            //this is true and therefore predictions are equal
            System.out.println("Predictions are equal to tensorflow");
        }
        //add to graph to demonstrate pytorch like capability
        System.out.println("Adding new op to graph..");
        SDVariable linspaceConstant = graphSD.var("linspace", Nd4j.linspace(1, 10, 10));
        SDVariable totalOutput = graphSD.getVariable("output").add(linspaceConstant);
        INDArray totalOutputArr = totalOutput.eval();
        System.out.println(totalOutputArr);

    }

    //A simple helper function to load the inputs and corresponding outputs generated from tensorflow
    //Two cases: {input_a,prediction_a} and {input_b,prediction_b}
    protected static Map<String, INDArray> readPlaceholdersAndPredictions() throws IOException {
        String[] toReadList = {"input_a", "input_b", "prediction_a", "prediction_b"};
        Map<String, INDArray> arraysFromPython = new HashMap<>();
        for (int i = 0; i < toReadList.length; i++) {
            String varShapePath = new ClassPathResource(BASE_DIR + "/" + toReadList[i] + ".shape").getFile().getPath();
            String varValuePath = new ClassPathResource(BASE_DIR + "/" + toReadList[i] + ".csv").getFile().getPath();
            int[] varShape = Nd4j.readNumpy(varShapePath, ",").data().asInt();
            float[] varContents = Nd4j.readNumpy(varValuePath).data().asFloat();
            arraysFromPython.put(toReadList[i], Nd4j.create(varContents).reshape(varShape));
        }
        return arraysFromPython;
    }
}
