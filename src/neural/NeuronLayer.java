package neural;

import activation.ActivationFunction;
import org.jblas.DoubleMatrix;

public class NeuronLayer {

    NeuronLayer previousLayer; // previous layer in the network
    NeuronLayer nextLayer; // next layers in the network

    DoubleMatrix input; // input of the layer
    DoubleMatrix weightedOutput; // weighted output of the layer
    DoubleMatrix output; // output of the layer

    DoubleMatrix weights; // current layers weights
    DoubleMatrix bias; // current layers biases

    ActivationFunction activation; // activation function used by all neurons in the layer
    int neuronCount; // amount of neurons in the layer
    boolean hasBias;

    /**
     * Create a Neural Network Layer containing the specified number of neurons and using the provided activation function.
     * @param activation Activation function that all neurons in the layer will be using.
     * @param neuronCount Count of neurons that are present in the layer.
     * @param hasBias Whether the layer has a bias neuron or not.
     */
    public NeuronLayer(ActivationFunction activation, int neuronCount, boolean hasBias) {
        // set the member variables
        this.activation = activation;
        this.neuronCount = neuronCount;
        this.hasBias = hasBias;
    }

    /**
     * Initialize the weights and bias using a Gaussian distribution with mean 0 and standard deviation 1 over the
     * square root of the number of weights connecting to the same neuron. Initialize the biases using a Gaussian
     * distribution with mean 0 and standard deviation 1.
     *
     * Note that the first layer is assumed to be an input layer, and by convention we won't set any biases for those
     * neurons, since biases are only ever used in computing the outputs from later layers.
     */
    void initialize() {
        // ignore initialization if we are the input layer
        if (previousLayer == null) {
            this.bias = DoubleMatrix.zeros(this.neuronCount);
            this.weights = null;
        } else {
            this.bias = this.hasBias ? DoubleMatrix.randn(this.neuronCount) : DoubleMatrix.zeros(this.neuronCount);
            this.weights = DoubleMatrix.randn(this.neuronCount, previousLayer.neuronCount).div(Math.sqrt(previousLayer.neuronCount));
        }
    }

    /**
     * Set the input for the layer.
     * @param input Input for the layer.
     */
    void setInput(DoubleMatrix input) {
        this.input = input;
    }

    /**
     * Return the input of the layer.
     * @return Input of the layer.
     */
    public DoubleMatrix getInput() {
        return this.input;
    }

    /**
     * Return the output of the layer.
     * @return Output of the layer.
     */
    public DoubleMatrix getOutput() {
        return this.output;
    }

    /**
     * Return the weighted output of the layer.
     * @return Derivative of the layer.
     */
    public DoubleMatrix getWeightedOutput() {
        return this.weightedOutput;
    }

    /**
     * Return the weights of the layer.
     * @return Weights of the layer.
     */
    public DoubleMatrix getWeights() {
        return this.weights;
    }

    /**
     * Sets the weights to the specified matrix.
     * @param weights New weight matrix.
     */
    public void setWeights(DoubleMatrix weights) {
        this.weights = weights;
    }

    /**
     * Return the biases of the layer.
     * @return Bias of the layer.
     */
    public DoubleMatrix getBias() {
        return this.bias;
    }

    /**
     * Sets the biases to the specified matrix.
     * @param bias New bias matrix.
     */
    public void setBias(DoubleMatrix bias) {
        this.bias = bias;
    }

    /**
     * Return the activation function used by the neurons in the layer.
     * @return Activation function used in the layer.
     */
    public ActivationFunction getActivationFunction() {
        return this.activation;
    }

    /**
     * Is the current layer the output layer.
     * @return True is layer is an output layer, false otherwise.
     */
    public boolean isOutputLayer() {
        return nextLayer == null;
    }

    /**
     * Is the current layer the input layer.
     * @return True is layer is an input layer, false otherwise.
     */
    public boolean isInputLayer() {
        return previousLayer == null;
    }

    /**
     * Return the previous layer.
     * @return Previous layer.
     */
    public NeuronLayer getPreviousLayer() {
        return previousLayer;
    }

    /**
     * Activate the neurons based on the input, set and feed the output of the layer to the next layers input (if any).
     */
    void feedForward() {
        if (isInputLayer()) {
            // input layer will only pass the input to the next layer
            this.output = input;
            this.weightedOutput = null;
        } else {
            this.weightedOutput = weights.mmul(input).add(bias); // n(l) = w(l) * a(l-1) + b(l)
            this.output = activation.fn(this.weightedOutput); // a(l) = f(n(l))
        }
        // check if we need to feed the result to the next layer
        if (!isOutputLayer()) {
            this.nextLayer.input = this.output;
            this.nextLayer.feedForward();
        }
    }
}
