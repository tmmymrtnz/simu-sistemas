package tp5.simulacion.engine;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.core.Vector2;
import tp5.simulacion.model.PedestrianDynamicsModel;
import tp5.simulacion.output.SimulationOutputWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Runs the pedestrian simulation using a velocity-Verlet (kick-drift-kick) scheme.
 * The engine remains model-agnostic: models provide accelerations and optional post-step hooks.
 */
public class SimulationEngine {
    private final SimulationState state;
    private final PedestrianDynamicsModel model;
    private final NeighbourProvider neighbourProvider;
    private final SimulationOutputWriter outputWriter;
    private final ContactTracker contactTracker = new ContactTracker();

    private final long outputStride;

    public SimulationEngine(
            SimulationState state,
            PedestrianDynamicsModel model,
            NeighbourProvider neighbourProvider,
            SimulationOutputWriter outputWriter
    ) {
        this.state = state;
        this.model = model;
        this.neighbourProvider = neighbourProvider;
        this.outputWriter = outputWriter;
        this.outputStride = computeStride(state.getConfig());
    }

    private long computeStride(SimulationConfig config) {
        double ratio = config.outputInterval() / config.timeStep();
        long stride = Math.round(ratio);
        if (Math.abs(ratio - stride) > 1e-9) {
            throw new IllegalArgumentException("outputInterval must be an integer multiple of timeStep");
        }
        return Math.max(1, stride);
    }

    public void run() throws IOException {
        List<Agent> agents = state.getMobileAgents();
        Map<Integer, Vector2> initialAccelerations = computeAccelerations(agents);
        applyAccelerations(agents, initialAccelerations);
        cachedAccelerations = buildAccelerationSnapshot(agents);
        writeStateSnapshot(); // initial condition

        while (state.getStep() < state.getConfig().totalSteps()) {
            stepOnce(agents);
            state.advanceStep();
            if ((state.getStep() % outputStride) == 0) {
                writeStateSnapshot();
            }
        }
    }

    private void stepOnce(List<Agent> agents) throws IOException {
        double dt = state.getConfig().timeStep();
        double halfDt = 0.5 * dt;
        double L = state.getConfig().domainSize();
        boolean periodic = state.getConfig().periodic();
        Agent central = state.getCentralAgent();

        Map<Integer, Vector2> halfStepVelocities = new HashMap<>();
        Map<Integer, Boolean> wrapped = new HashMap<>();

        // First kick + drift
        for (Agent agent : agents) {
            Vector2 acc = agent.getAcceleration();
            Vector2 velocity = agent.getVelocity();
            Vector2 vHalf = velocity.add(acc.multiply(halfDt));
            Vector2 newPosition = agent.getPosition().add(vHalf.multiply(dt));

            boolean wrap = false;
            if (periodic) {
                double nx = newPosition.x();
                double ny = newPosition.y();
                if (nx < 0) {
                    nx = (nx % L + L) % L;
                    wrap = true;
                } else if (nx >= L) {
                    nx = nx % L;
                    wrap = true;
                }
                if (ny < 0) {
                    ny = (ny % L + L) % L;
                    wrap = true;
                } else if (ny >= L) {
                    ny = ny % L;
                    wrap = true;
                }
                newPosition = new Vector2(nx, ny);
            }

            agent.setPosition(newPosition);
            agent.setVelocity(vHalf); // store half-step velocity until the second kick
            halfStepVelocities.put(agent.getId(), vHalf);
            wrapped.put(agent.getId(), wrap);
        }

        Map<Integer, Vector2> nextAccelerations = computeAccelerations(agents);

        long contactOrdinal = state.getUniqueContactCount();
        double eventTime = state.getTime() + dt;
        List<double[]> accForOutput = new ArrayList<>(agents.size());

        for (Agent agent : agents) {
            Vector2 vHalf = halfStepVelocities.getOrDefault(agent.getId(), agent.getVelocity());
            Vector2 nextAcc = nextAccelerations.getOrDefault(agent.getId(), Vector2.ZERO);
            Vector2 newVelocity = vHalf.add(nextAcc.multiply(halfDt));

            agent.setVelocity(newVelocity);
            agent.setAcceleration(nextAcc);

            boolean wasWrapped = wrapped.getOrDefault(agent.getId(), false);
            contactTracker.resetIfWrapped(agent, wasWrapped);
            if (central != null && contactTracker.shouldRegisterContact(agent, central.getPosition(), central.getRadius())) {
                contactOrdinal = state.incrementContactCount();
                outputWriter.writeContact(contactOrdinal, eventTime, agent.getId());
            }

            accForOutput.add(new double[]{nextAcc.x(), nextAcc.y()});
        }

        cachedAccelerations = accForOutput;
        model.postIntegration(state, dt);
    }

    private List<double[]> cachedAccelerations = new ArrayList<>();

    private void writeStateSnapshot() throws IOException {
        List<Agent> agents = state.getMobileAgents();
        if (cachedAccelerations.size() != agents.size()) {
            cachedAccelerations = new ArrayList<>(agents.size());
            for (Agent ignored : agents) {
                cachedAccelerations.add(new double[]{0.0, 0.0});
            }
        }
        outputWriter.writeState(state.getStep(), state.getTime(), agents, cachedAccelerations);
    }

    private Map<Integer, Vector2> computeAccelerations(List<Agent> agents) {
        Map<Integer, Set<Integer>> neighbourhood = neighbourProvider.compute(agents);
        return model.computeAccelerations(state, neighbourhood);
    }

    private void applyAccelerations(List<Agent> agents, Map<Integer, Vector2> accelerations) {
        for (Agent agent : agents) {
            Vector2 acc = accelerations.getOrDefault(agent.getId(), Vector2.ZERO);
            agent.setAcceleration(acc);
        }
    }

    private List<double[]> buildAccelerationSnapshot(List<Agent> agents) {
        List<double[]> snapshot = new ArrayList<>(agents.size());
        for (Agent agent : agents) {
            Vector2 acc = agent.getAcceleration();
            snapshot.add(new double[]{acc.x(), acc.y()});
        }
        return snapshot;
    }
}
