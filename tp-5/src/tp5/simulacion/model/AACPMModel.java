package tp5.simulacion.model;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.core.Vector2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implementation inspired by the Anisotropic Contractile Particle Model with Avoidance (Martin & Parisi, 2024).
 * Captures a contractile drive, anisotropic contact weighting, and a rule-based evasive manoeuvre.
 */
public class AACPMModel implements PedestrianDynamicsModel {

    public static class Parameters {
        public final double contractileK;
        public final double fieldOfViewExponent;
        public final double bodyK;
        public final double slidingKappa;
        public final double avoidanceGain;
        public final double avoidanceTimeThreshold;
        public final double avoidanceDelta;
        public final double omegaAlign;
        public final double desiredSpeed;
        public final double maxSpeed;
        public final double avoidanceSpeedScale;

        public Parameters(double contractileK,
                          double fieldOfViewExponent,
                          double bodyK,
                          double slidingKappa,
                          double avoidanceGain,
                          double avoidanceTimeThreshold,
                          double avoidanceDelta,
                          double omegaAlign,
                          double desiredSpeed,
                          double maxSpeed,
                          double avoidanceSpeedScale) {
            this.contractileK = contractileK;
            this.fieldOfViewExponent = fieldOfViewExponent;
            this.bodyK = bodyK;
            this.slidingKappa = slidingKappa;
            this.avoidanceGain = avoidanceGain;
            this.avoidanceTimeThreshold = avoidanceTimeThreshold;
            this.avoidanceDelta = avoidanceDelta;
            this.omegaAlign = omegaAlign;
            this.desiredSpeed = desiredSpeed;
            this.maxSpeed = maxSpeed;
            this.avoidanceSpeedScale = avoidanceSpeedScale;
        }
    }

    private final Parameters params;

    public AACPMModel(Parameters params) {
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
            Vector2 heading = agent.getHeading().normalized();
            Vector2 desiredDirection = agent.getDesiredDirection().normalized();
            double desiredSpeed = params.desiredSpeed * agent.getDesiredSpeed();

            AvoidanceResult avoidance = computeAvoidance(agent, agentById, central, L, periodic, neighbourhood);
            double effectiveDesiredSpeed = desiredSpeed;
            if (avoidance.triggered) {
                effectiveDesiredSpeed *= params.avoidanceSpeedScale;
            }

            Vector2 totalForce = desiredDirection.multiply(effectiveDesiredSpeed)
                    .subtract(agent.getVelocity())
                    .multiply(params.contractileK);

            Set<Integer> neighbours = neighbourhood.getOrDefault(agent.getId(), Set.of());
            for (Integer neighbourId : neighbours) {
                Agent other = agentById.get(neighbourId);
                if (other == null) continue;
                totalForce = totalForce.add(weightedContactForce(agent, other, heading, L, periodic));
            }

            if (central != null) {
                totalForce = totalForce.add(weightedContactForce(agent, central, heading, L, periodic));
            }

            if (avoidance.triggered) {
                totalForce = totalForce.add(avoidance.lateralForce);
            }

            Vector2 acceleration = totalForce.divide(agent.getMass());
            accelerations.put(agent.getId(), acceleration);
        }

        return accelerations;
    }

    private Vector2 weightedContactForce(Agent agent, Agent other, Vector2 heading, double L, boolean periodic) {
        Vector2 displacement = relativePosition(agent.getPosition(), other.getPosition(), L, periodic);
        double distance = displacement.norm();
        if (distance == 0) {
            displacement = new Vector2(1e-6, 0);
            distance = displacement.norm();
        }
        Vector2 n = displacement.divide(distance); // pointing from other to agent
        distance = Math.max(distance, 1e-6);

        double overlap = agent.getRadius() + other.getRadius() - distance;
        double g = Math.max(overlap, 0.0);
        if (g <= 0) {
            return Vector2.ZERO;
        }

        Vector2 t = new Vector2(-n.y(), n.x());
        Vector2 relativeVelocity = other.getVelocity().subtract(agent.getVelocity());
        double deltaTangential = relativeVelocity.dot(t);

        Vector2 body = n.multiply(params.bodyK * g);
        Vector2 friction = t.multiply(params.slidingKappa * g * deltaTangential);

        double cosTheta = clamp(heading.dot(n.normalized()), -1.0, 1.0);
        double weight = Math.max(0.0, Math.pow(Math.max(0.0, cosTheta), params.fieldOfViewExponent));
        return body.add(friction).multiply(weight);
    }

    private AvoidanceResult computeAvoidance(Agent agent,
                                             Map<Integer, Agent> agentById,
                                             Agent central,
                                             double L,
                                             boolean periodic,
                                             Map<Integer, Set<Integer>> neighbourhood) {
        double bestTTC = Double.POSITIVE_INFINITY;
        Vector2 bestDirection = Vector2.ZERO;

        Set<Integer> neighbourIds = neighbourhood.getOrDefault(agent.getId(), Set.of());
        for (Integer neighbourId : neighbourIds) {
            Agent other = agentById.get(neighbourId);
            if (other == null) continue;
            AvoidanceCandidate candidate = avoidanceCandidate(agent, other, L, periodic);
            if (candidate != null && candidate.ttc < bestTTC) {
                bestTTC = candidate.ttc;
                bestDirection = candidate.lateralDirection;
            }
        }

        if (central != null) {
            AvoidanceCandidate centralCandidate = avoidanceCandidate(agent, central, L, periodic);
            if (centralCandidate != null && centralCandidate.ttc < bestTTC) {
                bestTTC = centralCandidate.ttc;
                bestDirection = centralCandidate.lateralDirection;
            }
        }

        boolean triggered = bestTTC < params.avoidanceTimeThreshold;
        Vector2 lateral = triggered ? bestDirection.multiply(params.avoidanceGain) : Vector2.ZERO;

        return new AvoidanceResult(triggered, lateral);
    }

    private AvoidanceCandidate avoidanceCandidate(Agent agent, Agent other, double L, boolean periodic) {
        Vector2 relPos = relativePosition(agent.getPosition(), other.getPosition(), L, periodic);
        Vector2 relVel = agent.getVelocity().subtract(other.getVelocity());

        double radius = agent.getRadius() + other.getRadius() + params.avoidanceDelta;
        double a = relVel.normSq();
        double b = 2 * relPos.dot(relVel);
        double c = relPos.normSq() - radius * radius;

        if (a < 1e-8) {
            return null;
        }

        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return null;
        }

        double sqrt = Math.sqrt(discriminant);
        double t1 = (-b - sqrt) / (2 * a);
        double t2 = (-b + sqrt) / (2 * a);
        double ttc = Math.min(t1, t2);
        if (ttc <= 0) {
            ttc = Math.max(t1, t2);
        }
        if (ttc <= 0) {
            return null;
        }

        Vector2 heading = agent.getHeading().normalized();
        Vector2 left = new Vector2(-heading.y(), heading.x());
        Vector2 toOther = relPos.normalized();
        double cross = heading.x() * toOther.y() - heading.y() * toOther.x();
        double sign = cross >= 0 ? -1.0 : 1.0;
        Vector2 lateral = left.multiply(sign);

        return new AvoidanceCandidate(ttc, lateral.normalized());
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

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    @Override
    public void postIntegration(SimulationState state, double dt) {
        double dtAlign = params.omegaAlign * dt;
        for (Agent agent : state.getMobileAgents()) {
            Vector2 velocity = agent.getVelocity();
            double speed = velocity.norm();
            if (speed > params.maxSpeed) {
                Vector2 clamped = velocity.divide(speed).multiply(params.maxSpeed);
                agent.setVelocity(clamped);
                velocity = clamped;
                speed = params.maxSpeed;
            }

            Vector2 currentHeading = agent.getHeading().normalized();
            Vector2 target = speed > 1e-6 ? velocity.divide(speed) : currentHeading;
            Vector2 newHeading = currentHeading.add(target.subtract(currentHeading).multiply(dtAlign)).normalized();
            agent.setHeading(newHeading);
        }
    }

    @Override
    public List<String> describeParameters() {
        return List.of(
                "AACPM_kc=" + params.contractileK,
                "AACPM_beta=" + params.fieldOfViewExponent,
                "AACPM_k=" + params.bodyK,
                "AACPM_kappa=" + params.slidingKappa,
                "AACPM_kavo=" + params.avoidanceGain,
                "AACPM_tau=" + params.avoidanceTimeThreshold,
                "AACPM_delta=" + params.avoidanceDelta,
                "AACPM_omega=" + params.omegaAlign,
                "AACPM_vdes=" + params.desiredSpeed,
                "AACPM_vmax=" + params.maxSpeed,
                "AACPM_alpha=" + params.avoidanceSpeedScale
        );
    }

    private record AvoidanceResult(boolean triggered, Vector2 lateralForce) {}

    private record AvoidanceCandidate(double ttc, Vector2 lateralDirection) {}
}
