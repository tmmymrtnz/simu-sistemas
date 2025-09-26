package simulation;

import java.util.List;
import java.util.function.Function;

public class Beeman implements  NumericMethod{

    private double previousAcceleration;

    private final AccelerationFunction accelerationFunction;

    Beeman(AccelerationFunction accelerationFunction, double initialAcceleration) {
        this.previousAcceleration = initialAcceleration;
        this.accelerationFunction = accelerationFunction;
    }

    @Override
    public State solve(Double x, Double v, Double t, Double dt) {
        double a = accelerationFunction.computeAcceleration(x,v,t); // Acceleration at current time step
        double x1 = x + v * dt + (2.0/3.0) * a * dt * dt - (1.0/6.0) * previousAcceleration * dt * dt * (t-dt);// Acceleration at next time step

        double predictedVelocity = v + (3.0/2.0) * a * dt - (1.0/2.0) * previousAcceleration * dt; // Predicted velocity at next time step
        double a1 = accelerationFunction.computeAcceleration(x1, predictedVelocity, t + dt);

        double v1 = v + (1.0/3.0) * a1 * dt + (5.0/6.0) * a * dt - (1.0/6.0) * previousAcceleration * dt; // Velocity at next time step
        return new State(x1, v1);
    }
}
