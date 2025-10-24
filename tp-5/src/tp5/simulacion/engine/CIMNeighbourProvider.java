package tp5.simulacion.engine;

import tp5.simulacion.cim.CellIndexMethod;
import tp5.simulacion.core.Agent;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Adapter that exposes the Cell Index Method implementation via the NeighbourProvider interface.
 */
public class CIMNeighbourProvider implements NeighbourProvider {
    private final CellIndexMethod cim;

    public CIMNeighbourProvider(CellIndexMethod cim) {
        this.cim = cim;
    }

    @Override
    public Map<Integer, Set<Integer>> compute(List<Agent> agents) {
        return cim.neighbours(agents);
    }
}
