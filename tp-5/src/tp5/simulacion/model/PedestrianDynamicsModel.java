package tp5.simulacion.model;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationState;
import tp5.simulacion.core.Vector2;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Common API for the pedestrian dynamics models required by the TP.
 * Implementations (SFM without social force, AACPM, etc.) only need to provide accelerations.
 */
public interface PedestrianDynamicsModel {

    /**
     * Computes accelerations for each movable agent in-place.
     *
     * @param state current mutable simulation state
     * @param neighbourhood adjacency information keyed by agent id
     * @return a map with the newly computed accelerations for each agent
     */
    Map<Integer, Vector2> computeAccelerations(SimulationState state, Map<Integer, Set<Integer>> neighbourhood);

    /**
     * Hook that runs after positions/velocities have been updated for the current step.
     *
     * @param state current simulation state (already advanced in time by dt)
     * @param dt    integration timestep
     */
    default void postIntegration(SimulationState state, double dt) {
        // default no-op
    }

    /**
     * Convenience hook for model-specific metadata to be included in the output header.
     */
    default List<String> describeParameters() {
        return List.of();
    }
}
