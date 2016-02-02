package cost;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class CrossEntropyCostFunction implements CostFunction {

    public String getName() {
        return "Cross Entropy Cost Function";
    }

    @Override
    public double fn(DoubleMatrix a, DoubleMatrix y) {
        return (y.neg().mul(MatrixFunctions.log(a)).sub(DoubleMatrix.ones(y.rows, y.columns).sub(y).mul(MatrixFunctions.log(DoubleMatrix.ones(a.rows, a.columns).sub(a)))).div(50000)).sum();
    }

    @Override
    public DoubleMatrix delta(DoubleMatrix z, DoubleMatrix a, DoubleMatrix y) {
        return (a.sub(y));
    }
}
