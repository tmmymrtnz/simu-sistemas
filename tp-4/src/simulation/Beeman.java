package simulation;

import java.util.List;
import java.util.function.Function;

public class Beeman implements  NumericMethod{

    private double previousAcceleration;

    private final AccelerationFunction accelerationFunction;

    Beeman(AccelerationFunction accelerationFunction, double x0, double v0) {
        this.accelerationFunction = accelerationFunction;
        previousAcceleration = accelerationFunction.computeAcceleration(x0, v0, 0.0);
    }

    @Override
    public State solve(Double x, Double v, Double t, Double dt) {
        double a = accelerationFunction.computeAcceleration(x, v, t); // Current acceleration
        
        // Position: x(t + Δt) = x(t) + v(t)Δt + (2/3)a(t)Δt² - (1/6)a(t - Δt)Δt²
        double x1 = x + v * dt + (2.0/3.0) * a * dt * dt - (1.0/6.0) * previousAcceleration * dt * dt;
        
        // Predicted velocity: v(t + Δt)_predicted = v(t) + (3/2)a(t)Δt - (1/2)a(t - Δt)Δt
        double predictedVelocity = v + (3.0/2.0) * a * dt - (1.0/2.0) * previousAcceleration * dt;
        
        // Acceleration at next time step
        double a1 = accelerationFunction.computeAcceleration(x1, predictedVelocity, t + dt);
        
        // Corrected velocity: v(t + Δt) = v(t) + (1/3)a(t + Δt)Δt + (5/6)a(t)Δt - (1/6)a(t - Δt)Δt
        double v1 = v + (1.0/3.0) * a1 * dt + (5.0/6.0) * a * dt - (1.0/6.0) * previousAcceleration * dt;
        
        // Update previous acceleration for next iteration
        previousAcceleration = a;
        
        return new State(x1, v1);
    }
}
