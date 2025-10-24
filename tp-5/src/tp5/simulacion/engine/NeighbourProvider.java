package tp5.simulacion.engine;

import tp5.simulacion.core.Agent;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Strategy interface for neighbour detection. Allows swapping CIM or brute-force.
 */
public interface NeighbourProvider {
    Map<Integer, Set<Integer>> compute(List<Agent> agents);
}
