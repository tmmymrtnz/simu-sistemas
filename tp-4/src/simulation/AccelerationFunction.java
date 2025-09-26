package simulation;

@FunctionalInterface
public interface AccelerationFunction {
    Double computeAcceleration(Double x, Double v, Double t);
}
