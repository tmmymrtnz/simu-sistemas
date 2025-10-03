package simulation;

import java.util.List;

/**
 * Contract for N-body integrators that advance the particle system by one time step.
 */
public interface NBodyIntegrator {

    void step(List<Particle> particles, double dt, double softeningLength);
}
