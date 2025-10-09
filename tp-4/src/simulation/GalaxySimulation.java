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

    public void run(Path energyPath) throws IOException {
        run(energyPath, null);
    }

    public void run(Path energyPath, Path snapshotPath) throws IOException {
        Files.createDirectories(energyPath.getParent());
        if (snapshotPath != null) {
            Files.createDirectories(snapshotPath.getParent());
        }

        double time = 0.0;
        double nextOutputTime = snapshotInterval > 0.0 ? snapshotInterval : dt;
        boolean outputEveryStep = snapshotInterval <= 0.0;

        try (PrintWriter energyWriter = new PrintWriter(Files.newBufferedWriter(energyPath))) {
            PrintWriter snapshotWriter = null;
            try {
                if (description != null && !description.isBlank()) {
                    energyWriter.println(description);
                }
                energyWriter.println(String.format(Locale.US, "# dt=%.8f snapshot_dt=%.8f", dt, snapshotInterval));
                energyWriter.println("# columns: t\ttotal_energy\tr_hm");

                if (snapshotPath != null) {
                    snapshotWriter = new PrintWriter(Files.newBufferedWriter(snapshotPath));
                    if (description != null && !description.isBlank()) {
                        snapshotWriter.println(description);
                    }
                    snapshotWriter.println(String.format(Locale.US, "# dt=%.8f snapshot_dt=%.8f", dt, snapshotInterval));
                    snapshotWriter.println("# columns: t\tid\tx\ty\tz\tvx\tvy\tvz\tdist_rhm");
                }

                HalfMassInfo halfMassInfo = computeHalfMassInfo();
                writeEnergyRow(energyWriter, time, halfMassInfo.radius());
                writeSnapshotRows(snapshotWriter, time, halfMassInfo);

                while (time < tf) {
                    integrator.step(particles, dt, softeningLength);
                    time += dt;

                    if (outputEveryStep || time + 1e-12 >= nextOutputTime) {
                        halfMassInfo = computeHalfMassInfo();
                        writeEnergyRow(energyWriter, time, halfMassInfo.radius());
                        writeSnapshotRows(snapshotWriter, time, halfMassInfo);
                        if (!outputEveryStep) {
                            nextOutputTime += snapshotInterval;
                        }
                    }
                }
            } finally {
                if (snapshotWriter != null) {
                    snapshotWriter.close();
                }
            }
        }
    }

    private void writeEnergyRow(PrintWriter energyWriter, double time, double halfMassRadius) {
        if (energyWriter == null) {
            return;
        }
        EnergySnapshot energy = computeEnergy();
        energyWriter.printf(Locale.US, "%.8f\t%.12f\t%.12f%n", time, energy.getTotal(), halfMassRadius);
    }

    private void writeSnapshotRows(PrintWriter snapshotWriter, double time, HalfMassInfo halfMassInfo) {
        if (snapshotWriter == null) {
            return;
        }
        for (Particle particle : particles) {
            Vector3 position = particle.getPosition();
            Vector3 velocity = particle.getVelocity();
            double dx = position.getX() - halfMassInfo.cx();
            double dy = position.getY() - halfMassInfo.cy();
            double dz = position.getZ() - halfMassInfo.cz();
            double radialDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            double distanceFromHalfMass = Math.abs(radialDistance - halfMassInfo.radius());
            snapshotWriter.printf(
                    Locale.US,
                    "%.8f\t%d\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f%n",
                    time,
                    particle.getId(),
                    position.getX(),
                    position.getY(),
                    position.getZ(),
                    velocity.getX(),
                    velocity.getY(),
                    velocity.getZ(),
                    distanceFromHalfMass
            );
        }
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

    private HalfMassInfo computeHalfMassInfo() {
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
        for (int i = 0; i < n; i++) {
            Vector3 position = particles.get(i).getPosition();
            double dx = position.getX() - cx;
            double dy = position.getY() - cy;
            double dz = position.getZ() - cz;
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            distances[i] = dist;
        }
        Arrays.sort(distances);

        double target = 0.5 * n;
        if (target <= 1.0) {
            return new HalfMassInfo(distances[0], cx, cy, cz);
        }

        int lowerIndex = Math.max(0, (int) Math.floor(target - 1));
        int upperIndex = Math.min(n - 1, lowerIndex + 1);
        double lowerMass = lowerIndex + 1;
        double upperMass = upperIndex + 1;

        if (upperIndex == lowerIndex) {
            return new HalfMassInfo(distances[lowerIndex], cx, cy, cz);
        }

        double fraction = (target - lowerMass) / (upperMass - lowerMass);
        double radius = distances[lowerIndex] + fraction * (distances[upperIndex] - distances[lowerIndex]);
        return new HalfMassInfo(radius, cx, cy, cz);
    }

    private static final class HalfMassInfo {
        private final double radius;
        private final double cx;
        private final double cy;
        private final double cz;

        private HalfMassInfo(double radius, double cx, double cy, double cz) {
            this.radius = radius;
            this.cx = cx;
            this.cy = cy;
            this.cz = cz;
        }

        double radius() {
            return radius;
        }

        double cx() {
            return cx;
        }

        double cy() {
            return cy;
        }

        double cz() {
            return cz;
        }
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
