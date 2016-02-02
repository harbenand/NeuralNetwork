import activation.SigmoidActivationFunction;
import cost.CostFunction;
import cost.CrossEntropyCostFunction;
import cost.QuadraticCostFunction;
import mnist.MnistLoader;
import mnist.MnistSet;
import neural.NeuronLayer;
import neural.NeuronNetwork;
import org.jblas.DoubleMatrix;
import training.BackPropagation;

import java.io.IOException;
import java.util.List;

public class Main {

    // filenames for the training set
    static final String trainLabelFile = "data/train-labels-idx1-ubyte.dat";
    static final String trainImageFile = "data/train-images-idx3-ubyte.dat";
    // filenames for the test set
    static final String testLabelFile = "data/t10k-labels-idx1-ubyte.dat";
    static final String testImageFile = "data/t10k-images-idx3-ubyte.dat";
    private static double[] avg;
    private static int[] classification;
    private static double[] ep;

    public static void main(String[] args) {
        MnistLoader loader = new MnistLoader();

        // load the MNIST training data using the loader
        List<MnistSet> trainingSets = null;
        List<MnistSet> validationSets = null;
        List<MnistSet> testSets = null;

        try {
            trainingSets = loader.create(trainLabelFile, trainImageFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            testSets = loader.create(testLabelFile, testImageFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // assert that we have data
        assert trainingSets != null;
        assert testSets != null;

        // split the training set into training and validation set
        validationSets = trainingSets.subList(50000, 60000);
        trainingSets = trainingSets.subList(0, 50000);

        boolean validate = true;
        boolean training = true;

        double[] learningRateList = {0.2};
        double[] momentumList = {0.0};
        int[] numHiddenNeurons = {100};

        /*if (validate) {
            for (double learningRate : learningRateList) {
                for (double momentum : momentumList) {
                    for (int hiddenNeurons : numHiddenNeurons) {
                        // create neural network
                        NeuronNetwork network = new NeuronNetwork();

                        // add layers to the network
                        network.addLayer(new NeuronLayer(null, 28 * 28, false));
                        network.addLayer(new NeuronLayer(new SigmoidActivationFunction(), hiddenNeurons, true));
                        network.addLayer(new NeuronLayer(new SigmoidActivationFunction(), 10, true));

                        // finalize the network which creates the neurons in the layers
                        network.finalizeNetwork();

                        // create the trainer that will be doing the training
                        BackPropagation backPropagationTrainer =
                                new BackPropagation(network, new CrossEntropyCostFunction(), learningRate, momentum);

                        // perform the training using the validation set
                        backPropagationTrainer.train(validationSets);

                        // use the test set to test accuracy of the network
                        int classified = 0;
                        for (MnistSet testPair : testSets) {
                            DoubleMatrix result = network.feedForward(testPair.getInput());
                            if (result.argmax() == testPair.getOutput().argmax()) {
                                classified++;
                            }
                        }
                        System.out.println("Learning Rate: " + learningRate + " | Momentum: " + momentum + " | Hidden Neurons: " + hiddenNeurons);
                        System.out.println("Accuracy: " + classified + "/" + testSets.size());
                    }
                }
            }
        }*/

        double learningRate = 0.15;
        double momentum = 0.0;
        int hiddenNeurons = 100;
        int iterations = 10;
        CostFunction costFunction = new QuadraticCostFunction();

        if (training) {
            // create neural network
            NeuronNetwork network = new NeuronNetwork();

            // add layers to the network
            network.addLayer(new NeuronLayer(null, 28 * 28, false));
            network.addLayer(new NeuronLayer(new SigmoidActivationFunction(), hiddenNeurons, true));
            network.addLayer(new NeuronLayer(new SigmoidActivationFunction(), 10, true));

            // finalize the network which creates the neurons in the layers
            network.finalizeNetwork();

            // create the trainer that will be doing the training
            BackPropagation backPropagationTrainer =
                    new BackPropagation(network, costFunction, learningRate, momentum);

            //int c = 1;
            //ep = new double[50001*10];

            for (int i = 0; i < iterations; i++) {
                // perform the training using the validation set
                avg = backPropagationTrainer.train(trainingSets);

                // use the test set to test accuracy of the network
                int classified = 0;
                int count = 1;
                classification = new int[10001];
                for (MnistSet testPair : testSets) {
                    DoubleMatrix result = network.feedForward(testPair.getInput());
                    if (result.argmax() == testPair.getOutput().argmax()) {
                        classified++;
                    }
                    classification[count] = classified;
                    count++;
                }
                System.out.println("Epoch " + i);
                System.out.println("Learning Rate: " + learningRate + " | Momentum: " + momentum + " | Hidden Neurons: " + hiddenNeurons);
                System.out.println("Accuracy: " + classified + "/" + testSets.size());
                /*for(int j = 1; j < avg.length; j++) {
                    ep[c] = avg[j];
                    //System.out.println(ep[c]);
                    c++;
                }*/
            }

        }
        //XYLineChartExample.main(args);
    }

    public double[] average()
    {
        return ep;
    }

    public int[] classified()
    {
        return classification;
    }
}
