package tp5.simulacion.core;

import java.util.List;

/**
 * Mutable simulation state encapsulating all agents and bookkeeping counters.
 */
public class SimulationState {
    private final SimulationConfig config;
    private final List<Agent> mobileAgents;
    private final Agent centralAgent;
    private long step;
    private double time;
    private long uniqueContactCount;

    public SimulationState(SimulationConfig config, List<Agent> mobileAgents, Agent centralAgent) {
        this.config = config;
        this.mobileAgents = mobileAgents;
        this.centralAgent = centralAgent;
        this.step = 0;
        this.time = 0.0;
        this.uniqueContactCount = 0;
    }

    public SimulationConfig getConfig() {
        return config;
    }

    public List<Agent> getMobileAgents() {
        return mobileAgents;
    }

    public Agent getCentralAgent() {
        return centralAgent;
    }

    public long getStep() {
        return step;
    }

    public double getTime() {
        return time;
    }

    public void advanceStep() {
        step += 1;
        time = step * config.timeStep();
    }

    public long getUniqueContactCount() {
        return uniqueContactCount;
    }

    public long incrementContactCount() {
        uniqueContactCount++;
        return uniqueContactCount;
    }
}
