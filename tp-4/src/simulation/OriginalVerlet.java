package simulation;

public class OriginalVerlet implements  NumericMethod {

    @Override
    public Double solve(Double x0, Double v0, Double t, Double dt) {
        double a = -x0; // Assuming a simple harmonic oscillator with k/m = 1
        double x1 = x0 + v0 * dt + 0.5 * a * dt * dt; // Position at next time step
        double a1 = -x1; // Acceleration at next time step
        double v1 = v0 + 0.5 * (a + a1) * dt; // Velocity at next time step
        return x1; // Return the new position
    }
}
