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
import java.util.List;
import java.util.Map;

/**
 * Minimal example of how to wire the simulation pieces together.
 *
 * This main class is intentionally simple: for a full study consider wrapping it in a CLI that sweeps
 * over area fractions, seeds, etc.
 */
public final class Main {
    private Main() {}

    public static void main(String[] args) throws IOException {
        Map<String, String> options = parseArgs(args);
        if (options.containsKey("help")) {
            printUsage();
            return;
        }

        double L = parseDouble(options, "domain", 6.0);
        double rc = parseDouble(options, "rc", 0.5);
        double dt = parseDouble(options, "dt", 0.002);
        double dt2 = parseDouble(options, "outputInterval", 0.02);
        long steps = resolveSteps(options, dt, 10_000);
        int agentCount = parseInt(options, "agents", 30);
        long seed = parseLong(options, "seed", 1234L);
        double minRadius = parseDouble(options, "minRadius", 0.18);
        double maxRadius = parseDouble(options, "maxRadius", 0.21);

        String modelName = options.getOrDefault("model", "sfm").toLowerCase();
        double desiredSpeed = parseDouble(options, "desiredSpeed", 1.7);

        PedestrianDynamicsModel model = switch (modelName) {
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
        System.out.println("Writing simulation output to: " + outputDir.toAbsolutePath());

        try (SimulationOutputWriter writer = new SimulationOutputWriter(outputDir, config, model, metadata)) {
            SimulationEngine engine = new SimulationEngine(initialState, model, neighbours, writer);
            engine.run();
        }
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

    private static void printUsage() {
        System.out.println("Usage: java tp5.simulacion.Main [--key=value ...]");
        System.out.println("Recognized keys:");
        System.out.println("  --output=path             Output directory (default auto-generated).");
        System.out.println("  --agents=int              Number of mobile agents (default 30).");
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
        String dir = "output/" + modelName + "_n" + agents + "_seed" + seed + "_steps" + steps;
        return Path.of(dir);
    }
}
