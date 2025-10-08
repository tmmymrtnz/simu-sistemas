package simulation;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * CLI entrypoint for the galaxy formation simulations (System 2).
 */
public class GalaxyMain {

    public static void main(String[] args) throws IOException {
        Map<String, String> params = parseArgs(args);

        int n = Integer.parseInt(params.getOrDefault("--N", "500"));
        double dt = Double.parseDouble(params.getOrDefault("--dt", "0.002"));
        double tf = Double.parseDouble(params.getOrDefault("--tf", "10.0"));
        double softening = Double.parseDouble(params.getOrDefault("--h", "0.05"));
        double velocityMagnitude = Double.parseDouble(params.getOrDefault("--speed", "0.1"));
        int runs = Integer.parseInt(params.getOrDefault("--runs", "1"));
        double snapshotDt = Double.parseDouble(params.getOrDefault("--dt-output", "0.02"));
        long seed = Long.parseLong(params.getOrDefault("--seed", String.valueOf(System.currentTimeMillis())));
        boolean collision = params.containsKey("--collision");
        boolean dumpSnapshots = params.containsKey("--dump-snapshots");
        int clusterSize = Integer.parseInt(params.getOrDefault("--cluster-size", String.valueOf(Math.max(1, n / 2))));
        double dx = Double.parseDouble(params.getOrDefault("--dx", "4.0"));
        double dy = Double.parseDouble(params.getOrDefault("--dy", "0.5"));

        Path outputDir = Paths.get(params.getOrDefault("--output-dir", "src/out/galaxy"));
        String integratorName = "velocity-verlet";

        ExecutorService executor = Executors.newFixedThreadPool(5);
        try {
            Future<?>[] futures = new Future<?>[runs];
            for (int run = 0; run < runs; run++) {
                final int runIndex = run;
                futures[run] = executor.submit(() -> {
                    long runSeed = seed + runIndex;
                    List<Particle> particles;
                    String description;
                    if (collision) {
                        particles = InitialConditionsFactory.generateClusterCollision(clusterSize, dx, dy, velocityMagnitude, runSeed);
                        description = InitialConditionsFactory.describeConfiguration("cluster_collision", particles.size(), velocityMagnitude, softening, dt, snapshotDt, tf);
                    } else {
                        particles = InitialConditionsFactory.generateGaussianSystem(n, velocityMagnitude, runSeed);
                        description = InitialConditionsFactory.describeConfiguration("gaussian", particles.size(), velocityMagnitude, softening, dt, snapshotDt, tf);
                    }

                    GalaxySimulation simulation = new GalaxySimulation(
                            particles,
                            new VelocityVerletNBody(),
                            dt,
                            softening,
                            tf,
                            snapshotDt,
                            description + String.format(Locale.US, " integrator=%s run=%d seed=%d", integratorName, runIndex, runSeed)
                    );

                    String runLabel = collision
                            ? String.format(Locale.US, "cluster_%d_run%03d", particles.size(), runIndex)
                            : String.format(Locale.US, "gaussian_N%d_run%03d", particles.size(), runIndex);

                    Path energyPath = outputDir.resolve("energy").resolve(runLabel + "_energy.txt");
                    Path snapshotPath = dumpSnapshots
                            ? outputDir.resolve("snapshots").resolve(runLabel + "_state.txt")
                            : null;

                    System.out.printf(Locale.US, "Running %s (dt=%.5f, dt_out=%.5f, tf=%.2f, h=%.3f) -> %s%n",
                            runLabel, dt, snapshotDt, tf, softening, energyPath);
                    if (snapshotPath != null) {
                        System.out.printf(Locale.US, "   snapshots -> %s%n", snapshotPath);
                    }

                    simulation.run(energyPath, snapshotPath);
                    return null;
                });
            }

            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    throw new IOException("Parallel simulation run failed", e);
                }
            }
        } finally {
            executor.shutdown();
            try {
                executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while awaiting simulation completion", ie);
            }
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> params = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String key = args[i];
            if (key.startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    params.put(key, args[++i]);
                } else {
                    params.put(key, "true");
                }
            }
        }
        return params;
    }
}
