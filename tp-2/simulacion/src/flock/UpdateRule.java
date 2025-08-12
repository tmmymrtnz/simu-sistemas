package flock;

import java.util.List;
import java.util.Random;

public interface UpdateRule {
    /** Calcula nextTheta para cada agente (no actualiza posiciones). */
    void updateThetas(List<Agent> agents, Grid grid, double eta, Random rng, boolean includeSelf);
}
