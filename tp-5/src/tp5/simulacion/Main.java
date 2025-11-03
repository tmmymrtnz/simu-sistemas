package tp5.simulacion;

import tp5.simulacion.cim.CellIndexMethod;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.engine.CIMNeighbourProvider;
import tp5.simulacion.engine.NeighbourProvider;
import tp5.simulacion.engine.SimulationEngine;
import tp5.simulacion.model.AACPMModel;
import tp5.simulacion.model.PedestrianDynamicsModel;
import tp5.simulacion.model.SFM2000Model;
import tp5.simulacion.output.SimulationOutputWriter;
import tp5.simulacion.setup.SimulationFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Minimal example of how to wire the simulation pieces together.
 *
 * This main class is intentionally simple: for a full study consider wrapping it in a CLI that sweeps
 * over area fractions, seeds, etc.
 */
public final class Main {
    private Main() {}

    public static void main(String[] args) throws IOException, InterruptedException {
        Map<String, String> parsedOptions = parseArgs(args);
        if (parsedOptions.containsKey("help")) {
            printUsage();
            return;
        }

        Map<String, String> options = Map.copyOf(parsedOptions);

        List<Integer> agentCounts = parseIntList(options, "agentsList");
        if (agentCounts.isEmpty()) {
            agentCounts = parseIntList(options, "agents");
        }
        if (agentCounts.isEmpty()) {
            agentCounts = List.of(parseInt(options, "agents", 30));
        }

        List<Long> seedValues = parseLongList(options, "seedsList");
        if (seedValues.isEmpty()) {
            seedValues = parseLongList(options, "seeds");
        }
        if (seedValues.isEmpty()) {
            seedValues = List.of(parseLong(options, "seed", 1234L));
        }

        List<RunSpec> runs = new ArrayList<>();
        for (int agents : agentCounts) {
            for (long seed : seedValues) {
                runs.add(new RunSpec(agents, seed));
            }
        }

        if (runs.isEmpty()) {
            System.out.println("No simulation runs requested.");
            return;
        }

        if (runs.size() == 1) {
            RunSpec spec = runs.get(0);
            runSimulation(options, spec.agents, spec.seed);
            return;
        }

        if (options.containsKey("output")) {
            throw new IllegalArgumentException("--output cannot be combined with multiple runs; remove it or provide unique paths.");
        }

        Map<Integer, List<RunSpec>> grouped = new LinkedHashMap<>();
        for (RunSpec spec : runs) {
            grouped.computeIfAbsent(spec.agents, ignored -> new ArrayList<>()).add(spec);
        }

        for (Map.Entry<Integer, List<RunSpec>> entry : grouped.entrySet()) {
            int agents = entry.getKey();
            List<RunSpec> group = entry.getValue();
            if (group.isEmpty()) {
                continue;
            }
            if (group.size() == 1) {
                RunSpec spec = group.get(0);
                System.out.println("Running simulation for N=" + agents + " seed=" + spec.seed);
                runSimulation(options, spec.agents, spec.seed);
                continue;
            }

            int threadCount = resolveThreadCount(options, group.size());
            System.out.println(
                    "Launching " + group.size() + " simulations for N=" + agents + " using " + threadCount + " threads."
            );

            ExecutorService executor = Executors.newFixedThreadPool(threadCount);
            List<Future<?>> futures = new ArrayList<>();
            try {
                for (RunSpec spec : group) {
                    futures.add(executor.submit(() -> {
                        runSimulation(options, spec.agents, spec.seed);
                        return null;
                    }));
                }
                for (Future<?> future : futures) {
                    future.get();
                }
            } catch (ExecutionException e) {
                executor.shutdownNow();
                Throwable cause = e.getCause();
                if (cause instanceof IOException ioException) {
                    throw ioException;
                }
                if (cause instanceof RuntimeException runtimeException) {
                    throw runtimeException;
                }
                if (cause instanceof Error error) {
                    throw error;
                }
                throw new RuntimeException("Simulation task failed", cause);
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
                throw e;
            } finally {
                executor.shutdown();
            }
        }
    }

    private static void runSimulation(Map<String, String> options, int agentCount, long seed) throws IOException {
        double L = parseDouble(options, "domain", 6.0);
        double rc = parseDouble(options, "rc", 0.5);
        double dt = parseDouble(options, "dt", 0.002);
        double dt2 = parseDouble(options, "outputInterval", 0.02);
        long steps = resolveSteps(options, dt, 10_000);
        double minRadius = parseDouble(options, "minRadius", 0.18);
        double maxRadius = parseDouble(options, "maxRadius", 0.21);
        double desiredSpeed = parseDouble(options, "desiredSpeed", 1.7);

        String modelName = options.getOrDefault("model", "sfm").toLowerCase();
        PedestrianDynamicsModel model = createModel(options, modelName);

        SimulationConfig config = new SimulationConfig(L, true, rc, dt, dt2, steps);
        SimulationState initialState = SimulationFactory.randomPlacement(
                config,
                agentCount,
                minRadius,
                maxRadius,
                desiredSpeed,
                seed
        );

        int M = Math.max(1, (int) Math.floor(config.domainSize() / config.interactionRadius()));
        CellIndexMethod cim = new CellIndexMethod(config.domainSize(), config.interactionRadius(), M, config.periodic(), true);
        NeighbourProvider neighbours = new CIMNeighbourProvider(cim);

        double phi = SimulationFactory.computeAreaFraction(initialState.getMobileAgents(), config.domainSize());
        List<String> metadata = new ArrayList<>();
        metadata.add("model=" + modelName);
        metadata.add("agents=" + agentCount);
        metadata.add("seed=" + seed);
        metadata.add("desiredSpeed=" + desiredSpeed);
        metadata.add("minRadius=" + minRadius);
        metadata.add("maxRadius=" + maxRadius);
        metadata.add("phi=" + phi);

        Path outputDir = resolveOutput(options, modelName, agentCount, seed, steps);
        System.out.println("[" + Thread.currentThread().getName() + "] Writing simulation output to: " + outputDir.toAbsolutePath());

        try (SimulationOutputWriter writer = new SimulationOutputWriter(outputDir, config, model, metadata)) {
            SimulationEngine engine = new SimulationEngine(initialState, model, neighbours, writer);
            engine.run();
            System.out.println("[" + Thread.currentThread().getName() + "] Finished simulation for agents=" + agentCount + " seed=" + seed);
        }
    }

    private static PedestrianDynamicsModel createModel(Map<String, String> options, String modelName) {
        return switch (modelName) {
            case "sfm" -> {
                double sfmA = parseDouble(options, "sfmA", 2000.0);
                double sfmB = parseDouble(options, "sfmB", 0.08);
                double sfmK = parseDouble(options, "sfmK", 120000.0);
                double sfmKappa = parseDouble(options, "sfmKappa", 240000.0);
                double sfmTau = parseDouble(options, "sfmTau",
                        options.containsKey("relaxation") ? parseDouble(options, "relaxation", 0.5) : 0.5);
                yield new SFM2000Model(new SFM2000Model.Parameters(sfmA, sfmB, sfmK, sfmKappa, sfmTau, List.of()));
            }
            case "aacpm" -> {
                double kc = parseDouble(options, "aacpmKc", 2.0);
                double beta = parseDouble(options, "aacpmBeta", 2.0);
                double body = parseDouble(options, "aacpmK", 120000.0);
                double kappa = parseDouble(options, "aacpmKappa", 240000.0);
                double kAvoid = parseDouble(options, "aacpmKavo", 1.5);
                double tauAvoid = parseDouble(options, "aacpmTau", 1.0);
                double deltaAvoid = parseDouble(options, "aacpmDelta", 0.05);
                double omegaAlign = parseDouble(options, "aacpmOmega", 2.0);
                double vDes = parseDouble(options, "aacpmVdes", 1.0);
                double vMax = parseDouble(options, "aacpmVmax", 2.0);
                double alpha = parseDouble(options, "aacpmAlpha", 0.5);
                yield new AACPMModel(new AACPMModel.Parameters(
                        kc, beta, body, kappa, kAvoid, tauAvoid, deltaAvoid, omegaAlign, vDes, vMax, alpha
                ));
            }
            default -> throw new IllegalArgumentException("Unknown model: " + modelName);
        };
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> opts = new HashMap<>();
        for (String arg : args) {
            if ("--help".equals(arg) || "-h".equals(arg)) {
                opts.put("help", "true");
                continue;
            }
            if (!arg.startsWith("--") || !arg.contains("=")) {
                throw new IllegalArgumentException("Invalid argument: " + arg + ". Use --key=value format.");
            }
            int idx = arg.indexOf('=');
            String key = arg.substring(2, idx);
            String value = arg.substring(idx + 1);
            opts.put(key, value);
        }
        return opts;
    }

    private static List<Integer> parseIntList(Map<String, String> opts, String key) {
        String raw = opts.get(key);
        if (raw == null) {
            return List.of();
        }
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) {
            return List.of();
        }
        String[] tokens = trimmed.split("[,;\\s]+");
        List<Integer> values = new ArrayList<>();
        for (String token : tokens) {
            if (!token.isEmpty()) {
                values.add(Integer.parseInt(token));
            }
        }
        return values;
    }

    private static List<Long> parseLongList(Map<String, String> opts, String key) {
        String raw = opts.get(key);
        if (raw == null) {
            return List.of();
        }
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) {
            return List.of();
        }
        String[] tokens = trimmed.split("[,;\\s]+");
        List<Long> values = new ArrayList<>();
        for (String token : tokens) {
            if (!token.isEmpty()) {
                values.add(Long.parseLong(token));
            }
        }
        return values;
    }

    private static int resolveThreadCount(Map<String, String> options, int runCount) {
        int requested = parseInt(options, "threads", 5);
        if (requested <= 0) {
            requested = 1;
        }
        requested = Math.min(requested, 5);
        if (runCount > 0) {
            requested = Math.min(requested, runCount);
        }
        return Math.max(1, requested);
    }

    private static void printUsage() {
        System.out.println("Usage: java tp5.simulacion.Main [--key=value ...]");
        System.out.println("Recognized keys:");
        System.out.println("  --output=path             Output directory (default auto-generated).");
        System.out.println("  --agents=int              Number of mobile agents (default 30).");
        System.out.println("  --agentsList=vals         Comma/space separated agent counts for batch runs.");
        System.out.println("  --model=sfm|aacpm         Dynamics model (default sfm).");
        System.out.println("  --relaxation=double       (Deprecated) use --sfmTau instead.");
        System.out.println("  --domain=double           Square domain size L in meters (default 6.0).");
        System.out.println("  --rc=double               Interaction radius for CIM (default 0.5).");
        System.out.println("  --dt=double               Integration time step (default 0.002).");
        System.out.println("  --outputInterval=double   Interval for state dumps (default 0.02).");
        System.out.println("  --steps=long              Number of integration steps (default 10000).");
        System.out.println("  --duration=double         Alternative to steps; simulation time in seconds.");
        System.out.println("  --seed=long               Random seed for initial placement (default 1234).");
        System.out.println("  --minRadius=double        Minimum particle radius (default 0.18).");
        System.out.println("  --maxRadius=double        Maximum particle radius (default 0.21).");
        System.out.println("  --desiredSpeed=double     Desired speed vd (default 1.7).");
        System.out.println("  --sfmA=double             SFM social force amplitude (default 2000).");
        System.out.println("  --sfmB=double             SFM social force range (default 0.08).");
        System.out.println("  --sfmK=double             SFM body stiffness (default 120000).");
        System.out.println("  --sfmKappa=double         SFM sliding friction coefficient (default 240000).");
        System.out.println("  --sfmTau=double           SFM relaxation time (default 0.5).");
        System.out.println("  --aacpmKc=double          AACPM contractile gain k_c (default 2.0).");
        System.out.println("  --aacpmBeta=double        AACPM anisotropy exponent beta (default 2.0).");
        System.out.println("  --aacpmK=double           AACPM body stiffness (default 120000).");
        System.out.println("  --aacpmKappa=double       AACPM friction coefficient (default 240000).");
        System.out.println("  --aacpmKavo=double        AACPM avoidance gain (default 1.5).");
        System.out.println("  --aacpmTau=double         AACPM avoidance time threshold (default 1.0).");
        System.out.println("  --aacpmDelta=double       AACPM avoidance buffer delta (default 0.05).");
        System.out.println("  --aacpmOmega=double       AACPM heading alignment rate (default 2.0).");
        System.out.println("  --aacpmVdes=double        AACPM desired speed scaling (default 1.0).");
        System.out.println("  --aacpmVmax=double        AACPM max speed (default 2.0).");
        System.out.println("  --aacpmAlpha=double       AACPM avoidance speed multiplier (default 0.5).");
        System.out.println("  --seeds=vals              Comma/space separated seeds for batch runs.");
        System.out.println("  --seedsList=vals          Alias for --seeds.");
        System.out.println("  --threads=int             Maximum concurrent simulations (default 5).");
    }

    private static int parseInt(Map<String, String> opts, String key, int defaultValue) {
        return opts.containsKey(key) ? Integer.parseInt(opts.get(key)) : defaultValue;
    }

    private static long parseLong(Map<String, String> opts, String key, long defaultValue) {
        return opts.containsKey(key) ? Long.parseLong(opts.get(key)) : defaultValue;
    }

    private static double parseDouble(Map<String, String> opts, String key, double defaultValue) {
        return opts.containsKey(key) ? Double.parseDouble(opts.get(key)) : defaultValue;
    }

    private static long resolveSteps(Map<String, String> opts, double dt, long defaultSteps) {
        if (opts.containsKey("steps")) {
            return Long.parseLong(opts.get("steps"));
        }
        if (opts.containsKey("duration")) {
            double duration = Double.parseDouble(opts.get("duration"));
            long steps = Math.max(1, Math.round(duration / dt));
            return steps;
        }
        return defaultSteps;
    }

    private static Path resolveOutput(Map<String, String> opts, String modelName, int agents, long seed, long steps) {
        if (opts.containsKey("output")) {
            return Path.of(opts.get("output"));
        }
        String base = opts.getOrDefault("output-base", "output");
        Path basePath = Path.of(base);
        if (!basePath.isAbsolute()) {
            basePath = Path.of("").toAbsolutePath().resolve(basePath).normalize();
        }
        String dir = modelName + "_n" + agents + "_seed" + seed + "_steps" + steps;
        return basePath.resolve(dir);
    }

    private static final class RunSpec {
        final int agents;
        final long seed;

        RunSpec(int agents, long seed) {
            this.agents = agents;
            this.seed = seed;
        }
    }
}
