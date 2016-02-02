package cost;

import activation.SigmoidActivationFunction;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class QuadraticCostFunction implements CostFunction {

    SigmoidActivationFunction f = new SigmoidActivationFunction();

    public String getName() {
        return "Quadratic Cost Function";
    }

    @Override
    public double fn(DoubleMatrix a, DoubleMatrix y) {
        //return 0.5 * MatrixFunctions.pow(a.sub(y), 2).norm1();
        return 0.5 * Math.pow((a.sub(y)).norm1(), 2);   //MatrixFunctions.pow(a.sub(y), 2).norm1();
    }

    @Override
    public DoubleMatrix delta(DoubleMatrix z, DoubleMatrix a, DoubleMatrix y) {
        return (a.sub(y)).mul(f.delta(z));
    }
}
