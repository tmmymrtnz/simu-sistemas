package simulation;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Runs an N-body simulation and persists position/velocity snapshots and energy diagnostics.
 */
public class GalaxySimulation {
    private final List<Particle> particles;
    private final NBodyIntegrator integrator;
    private final double dt;
    private final double softeningLength;
    private final double tf;
    private final double snapshotInterval;
    private final String description;

    public GalaxySimulation(List<Particle> particles,
                            NBodyIntegrator integrator,
                            double dt,
                            double softeningLength,
                            double tf,
                            double snapshotInterval,
                            String description) {
        this.particles = particles;
        this.integrator = integrator;
        this.dt = dt;
        this.softeningLength = softeningLength;
        this.tf = tf;
        this.snapshotInterval = snapshotInterval;
        this.description = description;
    }

    public void run(Path energyPath, Path particlesPath) throws IOException {
        Files.createDirectories(energyPath.getParent());
        Files.createDirectories(particlesPath.getParent());

        double time = 0.0;
        double nextOutputTime = snapshotInterval > 0.0 ? snapshotInterval : dt;
        boolean outputEveryStep = snapshotInterval <= 0.0;

        try (PrintWriter energyWriter = new PrintWriter(Files.newBufferedWriter(energyPath));
            PrintWriter particlesWriter = new PrintWriter(Files.newBufferedWriter(particlesPath))) {
            
            if (description != null && !description.isBlank()) {
                energyWriter.println(description);
            }
            energyWriter.println(String.format(Locale.US, "# dt=%.8f snapshot_dt=%.8f", dt, snapshotInterval));
            energyWriter.println("# columns: t\ttotal_energy\tr_hm");

            writeEnergyRow(energyWriter, time);
            writeParticlesToFile(particlesWriter, time);

            while (time < tf) {
                integrator.step(particles, dt, softeningLength);
                time += dt;

                if (outputEveryStep || time + 1e-12 >= nextOutputTime) {
                    writeEnergyRow(energyWriter, time);
                    writeParticlesToFile(particlesWriter, time);
                    if (!outputEveryStep) {
                        nextOutputTime += snapshotInterval;
                    }
                }
            }
        }
    }

    private void writeEnergyRow(PrintWriter energyWriter, double time) {
        if (energyWriter == null) {
            return;
        }
        EnergySnapshot energy = computeEnergy();
        double halfMassRadius = computeHalfMassRadius();
        energyWriter.printf(Locale.US, "%.8f\t%.12f\t%.12f%n", time, energy.getTotal(), halfMassRadius);
    }

    private EnergySnapshot computeEnergy() {
        double kinetic = 0.0;
        double potential = 0.0;
        int n = particles.size();

        for (Particle particle : particles) {
            Vector3 velocity = particle.getVelocity();
            double speedSquared = velocity.normSquared();
            kinetic += 0.5 * particle.getMass() * speedSquared;
        }

        double softeningSquared = softeningLength * softeningLength;
        for (int i = 0; i < n; i++) {
            Vector3 ri = particles.get(i).getPosition();
            for (int j = i + 1; j < n; j++) {
                Vector3 rj = particles.get(j).getPosition();
                Vector3 rij = ri.subtract(rj);
                double distanceSquared = rij.normSquared();
                double distance = Math.sqrt(distanceSquared + softeningSquared);
                potential += -particles.get(i).getMass() * particles.get(j).getMass() / distance;
            }
        }

        return new EnergySnapshot(kinetic, potential);
    }

    private double computeHalfMassRadius() {
        int n = particles.size();
        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        for (Particle particle : particles) {
            Vector3 position = particle.getPosition();
            cx += position.getX();
            cy += position.getY();
            cz += position.getZ();
        }
        cx /= n;
        cy /= n;
        cz /= n;

        double[] distances = new double[n];
        double maxDistance = 0.0;
        for (int i = 0; i < n; i++) {
            Vector3 position = particles.get(i).getPosition();
            double dx = position.getX() - cx;
            double dy = position.getY() - cy;
            double dz = position.getZ() - cz;
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            distances[i] = dist;
            if (dist > maxDistance) {
                maxDistance = dist;
            }
        }
        Arrays.sort(distances);

        double target = 0.5 * n;
        if (target <= 1.0) {
            return distances[0];
        }

        int lowerIndex = Math.max(0, (int) Math.floor(target - 1));
        int upperIndex = Math.min(n - 1, lowerIndex + 1);
        double lowerMass = lowerIndex + 1;
        double upperMass = upperIndex + 1;

        if (upperIndex == lowerIndex) {
            return distances[lowerIndex];
        }

        double fraction = (target - lowerMass) / (upperMass - lowerMass);
        return distances[lowerIndex] + fraction * (distances[upperIndex] - distances[lowerIndex]);
    }

    public static List<Particle> deepCopy(List<Particle> particles) {
        List<Particle> copy = new ArrayList<>(particles.size());
        for (Particle particle : particles) {
            copy.add(new Particle(
                    particle.getId(),
                    particle.getMass(),
                    particle.getPosition(),
                    particle.getVelocity()
            ));
        }
        return copy;
    }

    private void writeParticlesToFile(PrintWriter particlesWriter, double time) {
        if (particlesWriter == null) {
            return;
        }
        
        particlesWriter.println(particles.size());
        particlesWriter.println(String.format(Locale.US, "t=%.8f", time));
        
        for (Particle p : particles) {
            Vector3 pos = p.getPosition();
            Vector3 vel = p.getVelocity();
            // Format: mass pos_x pos_y pos_z vel_x vel_y vel_z
            particlesWriter.printf(Locale.US, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f%n",
                p.getMass(), pos.getX(), pos.getY(), pos.getZ(), vel.getX(), vel.getY(), vel.getZ());
        }
    }
}
