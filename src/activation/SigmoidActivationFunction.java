package activation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SigmoidActivationFunction implements ActivationFunction{
    @Override
    public DoubleMatrix fn(DoubleMatrix input) {
        return MatrixFunctions.pow(MatrixFunctions.exp(input.neg()).add(1), -1);
    }

    @Override
    public DoubleMatrix delta(DoubleMatrix input) {
        return fn(input).mul(DoubleMatrix.ones(input.rows, input.columns).sub(fn(input)));
    }
}
