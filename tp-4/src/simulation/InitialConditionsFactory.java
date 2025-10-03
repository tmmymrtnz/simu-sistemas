package simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Builds initial particle configurations for the galaxy simulations.
 */
public final class InitialConditionsFactory {

    private InitialConditionsFactory() {
    }

    public static List<Particle> generateGaussianSystem(int n, double velocityMagnitude, long seed) {
        Random random = new Random(seed);
        List<Particle> particles = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Vector3 position = new Vector3(random.nextGaussian(), random.nextGaussian(), random.nextGaussian());
            Vector3 velocity = randomDirection(random).scale(velocityMagnitude);
            particles.add(new Particle(i, 1.0, position, velocity));
        }
        return particles;
    }

    public static List<Particle> generateClusterCollision(int particlesPerCluster, double dx, double dy, double velocityMagnitude, long seed) {
        Random random = new Random(seed);
        List<Particle> particles = new ArrayList<>(particlesPerCluster * 2);
        Vector3 clusterAOffset = new Vector3(-dx / 2.0, -dy / 2.0, 0.0);
        Vector3 clusterBOffset = new Vector3(dx / 2.0, dy / 2.0, 0.0);
        Vector3 clusterAVelocity = new Vector3(velocityMagnitude, 0.0, 0.0);
        Vector3 clusterBVelocity = new Vector3(-velocityMagnitude, 0.0, 0.0);

        for (int i = 0; i < particlesPerCluster; i++) {
            Vector3 randomPos = new Vector3(random.nextGaussian(), random.nextGaussian(), random.nextGaussian());
            particles.add(new Particle(i, 1.0, clusterAOffset.add(randomPos), clusterAVelocity));
        }
        for (int i = 0; i < particlesPerCluster; i++) {
            Vector3 randomPos = new Vector3(random.nextGaussian(), random.nextGaussian(), random.nextGaussian());
            particles.add(new Particle(particlesPerCluster + i, 1.0, clusterBOffset.add(randomPos), clusterBVelocity));
        }
        return particles;
    }

    private static Vector3 randomDirection(Random random) {
        double theta = 2.0 * Math.PI * random.nextDouble();
        double phi = Math.acos(2.0 * random.nextDouble() - 1.0);
        double sinPhi = Math.sin(phi);
        return new Vector3(
                sinPhi * Math.cos(theta),
                sinPhi * Math.sin(theta),
                Math.cos(phi)
        );
    }

    public static String describeConfiguration(String type,
                                               int n,
                                               double velocityMagnitude,
                                               double softening,
                                               double dt,
                                               double snapshotDt,
                                               double tf) {
        return String.format(Locale.US,
                "# type=%s N=%d |v|=%.5f h=%.5f dt=%.5f dt_output=%.5f tf=%.3f",
                type,
                n,
                velocityMagnitude,
                softening,
                dt,
                snapshotDt,
                tf);
    }
}
