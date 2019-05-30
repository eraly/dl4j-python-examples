package org.deeplearning4j.modelimport.tensorflow;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Shows tensorflow import using mnist.
 * For the trained graph, please look at the python files under modelimport/tensorflow/
 * <p>
 * generate_model.py was used to train the graph
 *
 * @author Fariz Rahman
 * @author susaneraly
 */
public class MnistImportWithImages {
    private static SameDiff sd;

    public static void loadModel(String filepath) throws Exception {
        File file = new File(filepath);
        if (!file.exists()) {
            file = new ClassPathResource(filepath).getFile();
        }

        sd = TFGraphMapper.getInstance().importGraph(file);

        if (sd == null) {
            throw new Exception("Error loading model : " + file);
        }
    }

    public static INDArray predict(String filepath) throws IOException {
        File file = new File(filepath);
        if (!file.exists()) {
            file = new ClassPathResource(filepath).getFile();
        }

        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        INDArray image = loader.asMatrix(file);
        image.divi(255.);

        image = image.reshape(1, 28 * 28);
        sd.associateArrayWithVariable(image, sd.variableMap().get("input"));
        INDArray output = sd.execAndEndResult().get(NDArrayIndex.point(0));
        System.out.println(Arrays.toString(output.reshape(10).toDoubleVector()));
        return output;

    }

    public static int predictionToLabel(INDArray prediction) {
        return Nd4j.argMax(prediction.reshape(10)).getInt(0);
    }


    public static void main(String[] args) throws Exception {
        loadModel("modelimport/tensorflow/frozen_model.pb");
        for (int i = 1; i < 11; i++) {
            String file = "modelimport/tensorflow/images/img_%d.jpg";
            file = String.format(file, i);
            INDArray prediction = predict(file);
            int label = predictionToLabel(prediction);
            System.out.println(file + "  ===>  " + label);
        }

    }
}
