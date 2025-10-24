package tp5.simulacion.model;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.core.Vector2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implementation of the Social Force Model (Helbing-Farkas-Vicsek 2000).
 * Includes driving, social repulsion, body forces, and friction forces against other agents and walls.
 */
public class SFM2000Model implements PedestrianDynamicsModel {

    public static class Wall {
        private final Vector2 point;
        private final Vector2 normal;
        private final Vector2 tangent;

        public Wall(Vector2 point, Vector2 normal) {
            this.point = point;
            this.normal = normal.normalized();
            this.tangent = new Vector2(-this.normal.y(), this.normal.x());
        }

        public Vector2 getPoint() {
            return point;
        }

        public Vector2 getNormal() {
            return normal;
        }

        public Vector2 getTangent() {
            return tangent;
        }
    }

    public static class Parameters {
        public final double bodyK;
        public final double slidingKappa;
        public final double relaxationTime;
        public final List<Wall> walls;

        public Parameters(double socialA,
                          double socialB,
                          double bodyK,
                          double slidingKappa,
                          double relaxationTime,
                          List<Wall> walls) {
            this.bodyK = bodyK;
            this.slidingKappa = slidingKappa;
            this.relaxationTime = relaxationTime;
            this.walls = walls != null ? new ArrayList<>(walls) : List.of();
        }
    }

    private final Parameters params;

    public SFM2000Model(Parameters params) {
        this.params = params;
    }

    public Parameters getParams() {
        return params;
    }

    @Override
    public Map<Integer, Vector2> computeAccelerations(SimulationState state, Map<Integer, Set<Integer>> neighbourhood) {
        SimulationConfig config = state.getConfig();
        double L = config.domainSize();
        boolean periodic = config.periodic();
        List<Agent> agents = state.getMobileAgents();
        Agent central = state.getCentralAgent();

        Map<Integer, Agent> agentById = new HashMap<>();
        for (Agent agent : agents) {
            agentById.put(agent.getId(), agent);
        }

        Map<Integer, Vector2> accelerations = new HashMap<>();

        for (Agent agent : agents) {
            Vector2 totalForce = computeDrivingForce(agent);

            Set<Integer> neighbours = neighbourhood.getOrDefault(agent.getId(), Set.of());
            for (Integer neighbourId : neighbours) {
                Agent other = agentById.get(neighbourId);
                if (other == null) continue;
                totalForce = totalForce.add(interAgentForces(agent, other, L, periodic));
            }

            if (central != null) {
                totalForce = totalForce.add(interAgentForces(agent, central, L, periodic));
            }

            for (Wall wall : params.walls) {
                totalForce = totalForce.add(interWallForces(agent, wall));
            }

            Vector2 acceleration = totalForce.divide(agent.getMass());
            accelerations.put(agent.getId(), acceleration);
        }
        return accelerations;
    }

    private Vector2 computeDrivingForce(Agent agent) {
        Vector2 desiredVelocity = agent.getDesiredDirection().multiply(agent.getDesiredSpeed());
        return desiredVelocity.subtract(agent.getVelocity()).multiply(agent.getMass() / params.relaxationTime);
    }

    private Vector2 interAgentForces(Agent agent, Agent other, double L, boolean periodic) {
        Vector2 displacement = relativePosition(agent.getPosition(), other.getPosition(), L, periodic);
        double distance = displacement.norm();
        if (distance == 0) {
            // Overlapping centers, push in random tangent to avoid NaN
            displacement = new Vector2(1e-6, 0);
            distance = displacement.norm();
        }
        Vector2 n = displacement.divide(distance); // from other to agent
        Vector2 t = new Vector2(-n.y(), n.x());

        double overlap = agent.getRadius() + other.getRadius() - distance;
        double g = Math.max(overlap, 0.0);

        Vector2 body = n.multiply(params.bodyK * g);

        Vector2 relativeVelocity = other.getVelocity().subtract(agent.getVelocity());
        double deltaTangential = relativeVelocity.dot(t);
        Vector2 friction = t.multiply(params.slidingKappa * g * deltaTangential);

        return body.add(friction);
    }

    private Vector2 interWallForces(Agent agent, Wall wall) {
        Vector2 r = agent.getPosition().subtract(wall.getPoint());
        double distance = r.dot(wall.getNormal());
        Vector2 n = wall.getNormal();
        Vector2 t = wall.getTangent();

        double overlap = Math.max(agent.getRadius() - distance, 0.0);

        Vector2 body = n.multiply(params.bodyK * overlap);
        double tangentialVelocity = agent.getVelocity().dot(t);
        Vector2 friction = t.multiply(-params.slidingKappa * overlap * tangentialVelocity);
        return body.add(friction);
    }

    private static Vector2 relativePosition(Vector2 posA, Vector2 posB, double L, boolean periodic) {
        double dx = posA.x() - posB.x();
        double dy = posA.y() - posB.y();
        if (periodic) {
            if (dx > L / 2) dx -= L;
            if (dx < -L / 2) dx += L;
            if (dy > L / 2) dy -= L;
            if (dy < -L / 2) dy += L;
        }
        return new Vector2(dx, dy);
    }

    @Override
    public List<String> describeParameters() {
        return List.of(
                "SFM_k=" + params.bodyK,
                "SFM_kappa=" + params.slidingKappa,
                "SFM_tau=" + params.relaxationTime
        );
    }
}
