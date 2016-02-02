package activation;


import org.jblas.DoubleMatrix;

public interface ActivationFunction {
    DoubleMatrix fn(DoubleMatrix input);
    DoubleMatrix delta(DoubleMatrix input);
}
