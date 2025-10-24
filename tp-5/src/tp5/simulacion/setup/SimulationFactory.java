package tp5.simulacion.setup;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.core.Vector2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Utility to build initial simulation states using different placement strategies.
 */
public final class SimulationFactory {
    private SimulationFactory() {
    }

    /**
     * Generates a random (non-overlapping) placement of mobile agents within the square with periodic
     * boundaries. A simple rejection strategy is used; for dense systems this may need to be replaced
     * by a smarter packer.
     */
    public static SimulationState randomPlacement(
            SimulationConfig config,
            int count,
            double minRadius,
            double maxRadius,
            double desiredSpeed,
            long seed
    ) {
        Random rng = new Random(seed);
        double L = config.domainSize();

        Agent central = new Agent(
                false,
                0.21,
                80,
                new Vector2(L / 2.0, L / 2.0),
                Vector2.ZERO,
                Vector2.ZERO,
                0.0
        );

        List<Agent> agents = new ArrayList<>();
        int maxAttempts = 50_000;
        for (int i = 0; i < count; i++) {
            boolean placed = false;
            for (int attempt = 0; attempt < maxAttempts; attempt++) {
                double radius = minRadius + rng.nextDouble() * (maxRadius - minRadius);
                Vector2 position = new Vector2(rng.nextDouble() * L, rng.nextDouble() * L);

                if (overlaps(position, radius, central)) {
                    continue;
                }
                if (overlapsAny(position, radius, agents)) {
                    continue;
                }

                double angle = rng.nextDouble() * 2 * Math.PI;
                Vector2 desiredDir = new Vector2(Math.cos(angle), Math.sin(angle));
                Agent agent = new Agent(
                        true,
                        radius,
                        80,
                        position,
                        Vector2.ZERO,
                        desiredDir,
                        desiredSpeed
                );
                agents.add(agent);
                placed = true;
                break;
            }
            if (!placed) {
                throw new IllegalStateException("Failed to place all agents without overlap; decrease density or improve placement");
            }
        }
        return new SimulationState(config, agents, central);
    }

    public static double computeAreaFraction(List<Agent> agents, double domainSize) {
        double sum = 0.0;
        for (Agent agent : agents) {
            sum += Math.PI * agent.getRadius() * agent.getRadius();
        }
        return sum / (domainSize * domainSize);
    }

    private static boolean overlaps(Vector2 candidate, double radius, Agent other) {
        double dx = candidate.x() - other.getPosition().x();
        double dy = candidate.y() - other.getPosition().y();
        double distSq = dx * dx + dy * dy;
        double thr = radius + other.getRadius();
        return distSq <= thr * thr;
    }

    private static boolean overlapsAny(Vector2 candidate, double radius, List<Agent> agents) {
        for (Agent other : agents) {
            if (overlaps(candidate, radius, other)) {
                return true;
            }
        }
        return false;
    }
}
