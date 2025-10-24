package tp5.simulacion.core;

/**
 * Immutable configuration data for a simulation run.
 */
public record SimulationConfig(
        double domainSize,      // Square domain side (L)
        boolean periodic,
        double interactionRadius,
        double timeStep,
        double outputInterval,
        long totalSteps
) {
    public SimulationConfig {
        if (domainSize <= 0) throw new IllegalArgumentException("domainSize must be positive");
        if (interactionRadius <= 0) throw new IllegalArgumentException("interactionRadius must be positive");
        if (timeStep <= 0) throw new IllegalArgumentException("timeStep must be positive");
        if (outputInterval < timeStep) throw new IllegalArgumentException("outputInterval must be >= timeStep");
        if (totalSteps <= 0) throw new IllegalArgumentException("totalSteps must be positive");
    }
}
