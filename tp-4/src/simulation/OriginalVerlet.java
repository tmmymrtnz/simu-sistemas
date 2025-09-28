package simulation;

public class OriginalVerlet implements  NumericMethod {

    private double previousPosition;
    private final AccelerationFunction accelerationFunction;
    private boolean firstStep;

    OriginalVerlet(AccelerationFunction accelerationFunction, double x0) {
        this.accelerationFunction = accelerationFunction;
        this.previousPosition = x0;
        this.firstStep = true;
    }

    @Override
    public State solve(Double x0, Double v0, Double t, Double dt) {
        double a = accelerationFunction.computeAcceleration(x0, v0, t);
        double x1;
        
        if (firstStep) {
            // For first step, use: x(t + dt) = x(t) + v(t)*dt + (1/2)*a(t)*dt²
            x1 = x0 + v0 * dt + 0.5 * a * dt * dt;
            firstStep = false;
        } else {
            // Verlet formula: x(t + dt) = 2*x(t) - x(t - dt) + a(t)*dt²
            x1 = 2 * x0 - previousPosition + a * dt * dt;
        }
        
        // Velocity using: v(t) = [x(t + dt) - x(t - dt)] / (2*dt)
        double v1 = (x1 - previousPosition) / (2 * dt);
        
        // Update previous position for next iteration
        previousPosition = x0;
        
        return new State(x1, v1);
    }
}
