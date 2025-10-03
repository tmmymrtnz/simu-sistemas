package simulation;

import java.util.ArrayList;
import java.util.List;

/**
 * Velocity Verlet integrator for the softened gravitational N-body system.
 */
public class VelocityVerletNBody implements NBodyIntegrator {

    @Override
    public void step(List<Particle> particles, double dt, double softeningLength) {
        int n = particles.size();
        List<Vector3> accelerations = computeAccelerations(particles, softeningLength);

        List<Vector3> halfStepVelocities = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Particle particle = particles.get(i);
            Vector3 currentAcceleration = accelerations.get(i);
            Vector3 newPosition = particle.getPosition()
                    .add(particle.getVelocity().scale(dt))
                    .add(currentAcceleration.scale(0.5 * dt * dt));
            Vector3 halfStepVelocity = particle.getVelocity().add(currentAcceleration.scale(0.5 * dt));
            particle.setPosition(newPosition);
            halfStepVelocities.add(halfStepVelocity);
        }

        List<Vector3> nextAccelerations = computeAccelerations(particles, softeningLength);
        for (int i = 0; i < n; i++) {
            Particle particle = particles.get(i);
            Vector3 newVelocity = halfStepVelocities.get(i)
                    .add(nextAccelerations.get(i).scale(0.5 * dt));
            particle.setVelocity(newVelocity);
        }
    }

    private List<Vector3> computeAccelerations(List<Particle> particles, double softeningLength) {
        int n = particles.size();
        List<Vector3> accelerations = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            accelerations.add(Vector3.zero());
        }

        double softeningSquared = softeningLength * softeningLength;
        for (int i = 0; i < n; i++) {
            Particle pi = particles.get(i);
            Vector3 ri = pi.getPosition();
            for (int j = i + 1; j < n; j++) {
                Particle pj = particles.get(j);
                Vector3 rj = pj.getPosition();
                Vector3 rij = rj.subtract(ri);
                double distanceSquared = rij.normSquared();
                double base = distanceSquared + softeningSquared;
                double denom = Math.sqrt(base) * base;
                if (denom == 0.0) {
                    continue;
                }
                Vector3 accelerationContribution = rij.scale(1.0 / denom);
                accelerations.set(i, accelerations.get(i).add(accelerationContribution));
                accelerations.set(j, accelerations.get(j).add(accelerationContribution.scale(-1.0)));
            }
        }
        return accelerations;
    }
}
