package training;

import cost.CostFunction;
import mnist.MnistSet;
import neural.NeuronLayer;
import neural.NeuronNetwork;
import org.jblas.DoubleMatrix;

import java.util.List;

/**
 * Training class which performs training on a Neural Network using Back Propagation.
 */
public class BackPropagation {

    /**
     * Class which is going to store the weight deltas for momentum to work.
     */
    class WeightState {
        /**
         * Creates a state which holds the weight deltas.
         * @param prevOutputWeightGradient Output weight deltas.
         * @param prevHiddenWeightGradient Hidden weight deltas.
         */
        WeightState (DoubleMatrix prevOutputWeightGradient, DoubleMatrix prevHiddenWeightGradient) {
            this.prevOutputWeightGradient = prevOutputWeightGradient;
            this.prevHiddenWeightGradient = prevHiddenWeightGradient;
        }
        // previous weights that will be used for momentum
        DoubleMatrix prevOutputWeightGradient;
        DoubleMatrix prevHiddenWeightGradient;
    }

    private double learningRate;
    private double momentum;
    private NeuronNetwork network;
    private CostFunction costFunction;
    private WeightState prevState;
    private double[] avg;

    public BackPropagation(NeuronNetwork network, CostFunction costFunction, double learningRate, double momentum) {
        // set the class variables
        this.network = network;
        this.costFunction = costFunction;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.prevState = null;
        // notify the function that is going to be used
        System.out.println("Running BackPropagation with the following cost function: " + costFunction.getName());
    }


    public double[] train(List<MnistSet> trainingSets) {
        boolean output = false;
        int epoch = 1;
        double sum = 0;
        double average = 0.0;
        avg = new double[50002];

        for (MnistSet trainingSet : trainingSets) {
            double error = iteration(trainingSet);
            sum += error;
            average = sum/epoch;
            if (output) System.out.println("Epoch " + epoch + " | Error: " + error + " | Average: " + average);
            epoch++;
            avg[epoch] = average;
        }

        return avg;
    }

    public double iteration(MnistSet trainingPair) {
        DoubleMatrix delta;

        /**
         * Step 1: Set the corresponding activation for the neural network.
         */
        DoubleMatrix activation = trainingPair.getInput();

        /**
         * Step 2: Feedforward which will compute z and a for each layer in the network.
         *  z = weighedOutput
         *  a = output
         */
        DoubleMatrix result = this.network.feedForward(activation);

        /**
         * Step 3: Compute the output error vector.
         * This also does Step 5 when calculating 'outputWeightGradient'.
         */
        NeuronLayer outputLayer = this.network.getOutputLayer();
        delta = this.costFunction.delta(outputLayer.getWeightedOutput(), outputLayer.getOutput(), trainingPair.getOutput());
        // set the gradients
        DoubleMatrix outputWeightGradient = delta.mmul(outputLayer.getInput().transpose());
        DoubleMatrix outputBiasGradient = delta;

        /**
         * Step 4: Backpropagate the error to the hidden layer.
         * This also does Step 5 when calculating 'hiddenWeightGradient'.
         */
        NeuronLayer hiddenLayer = outputLayer.getPreviousLayer();
        // separating the calculation into two separate lines
        delta = outputLayer.getWeights().transpose().mmul(delta);
        delta = delta.mul(hiddenLayer.getActivationFunction().delta(hiddenLayer.getWeightedOutput()));
        // set the gradients
        DoubleMatrix hiddenWeightGradient = delta.mmul(hiddenLayer.getInput().transpose());
        DoubleMatrix hiddenBiasGradient = delta;

        /**
         * Apply the changes to the weights and bias for the output layer.
         */
        outputLayer.setWeights(
                (this.prevState == null || this.momentum == 0.0) ?
                        outputLayer.getWeights().sub(outputWeightGradient.mul(this.learningRate)) :
                        outputLayer.getWeights().sub(outputWeightGradient.mul(this.learningRate)).add(this.prevState.prevOutputWeightGradient.mul(this.momentum))
        );
        outputLayer.setBias(
                outputLayer.getBias().sub(outputBiasGradient.mul(this.learningRate))
        );

        /**
         * Apply the changes to the weights ans bias for the hidden layer.
         */
        hiddenLayer.setWeights(
                (this.prevState == null || this.momentum == 0.0) ?
                        hiddenLayer.getWeights().sub(hiddenWeightGradient.mul(this.learningRate)) :
                        hiddenLayer.getWeights().sub(hiddenWeightGradient.mul(this.learningRate)).add(this.prevState.prevHiddenWeightGradient.mul(this.momentum))
        );
        hiddenLayer.setBias(
                hiddenLayer.getBias().sub(hiddenBiasGradient.mul(this.learningRate))
        );

        /**
         * Update the state with the current weights.
         */
        this.prevState = new WeightState(outputWeightGradient, hiddenWeightGradient);

        /**
         * Return the error calculated using the cost function.
         */
        return this.costFunction.fn(result, trainingPair.getOutput());
    }
}
