package cost;

import org.jblas.DoubleMatrix;

public interface CostFunction {
    String getName();
    double fn(DoubleMatrix a, DoubleMatrix y);
    DoubleMatrix delta(DoubleMatrix z, DoubleMatrix a, DoubleMatrix y);
}
