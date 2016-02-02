package mnist;

import org.jblas.DoubleMatrix;

public class MnistSet {
    private final DoubleMatrix input; // input of the training set
    private final DoubleMatrix output; // expected output given the input

    public MnistSet(DoubleMatrix input, DoubleMatrix output) {
        this.input = input;
        this.output = output;
    }

    public DoubleMatrix getInput() {
        return this.input;
    }

    public DoubleMatrix getOutput() {
        return this.output;
    }
}
