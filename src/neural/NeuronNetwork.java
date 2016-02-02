package neural;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a neural network.
 */
public class NeuronNetwork {

    private NeuronLayer inputLayer; // input layer of the network
    private NeuronLayer outputLayer; // output layer of the network

    private List<NeuronLayer> layers = new ArrayList<>(); // List of layers in the network

    /**
     * Creates the Neural Network object.
     */
    public NeuronNetwork(){}

    /**
     * Adds the given layer to the current network.
     * @param layer Layer to be added.
     */
    public void addLayer(NeuronLayer layer) {
        this.layers.add(layer);

        // first layer that is added is the input layer
        if (layers.size() == 1) {
            inputLayer = layer;
        }

        // link the new layer to existing layers
        if (layers.size() > 1) {
            NeuronLayer previousLayer = layers.get(layers.size() - 2);
            previousLayer.nextLayer = layer;
            layer.previousLayer = previousLayer;
        }

        // update the output layer
        outputLayer = layers.get(layers.size() - 1);
    }

    /**
     * Finalize the network, which initializes the weights and biases on the layers.
     */
    public void finalizeNetwork() {
        this.layers.forEach(NeuronLayer::initialize);
    }

    /**
     * Returns the output layer of the network.
     * @return Output layer.
     */
    public NeuronLayer getOutputLayer() {
        return outputLayer;
    }

    /**
     * Given the input, it is fed to the network and the output is returned.
     * @param input Input that will be fed to the network.
     * @return Output of the neural network.
     */
    public DoubleMatrix feedForward(DoubleMatrix input) {
        // set the input on the input layer
        inputLayer.setInput(input);
        // evaluate the network
        layers.forEach(NeuronLayer::feedForward);
        return outputLayer.getOutput();
    }

}
