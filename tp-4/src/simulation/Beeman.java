package simulation;

public class Beeman  implements  NumericMethod{

    @Override
    public Double solve(Double x0, Double v0, Double t, Double dt) {
        double a0 = -x0; // Assuming a simple harmonic oscillator with k/m = 1
        double x1 = x0 + v0 * dt + (2.0/3.0) * a0 * dt * dt; // Position at next time step
        double a1 = -x1; // Acceleration at next time step
        double v1 = v0 + (1.0/3.0) * a1 * dt + (5.0/6.0) * a0 * dt; // Velocity at next time step
        return x1; // Return the new position
    }
}
