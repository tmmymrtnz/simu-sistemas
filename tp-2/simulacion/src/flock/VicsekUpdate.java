package flock;

import java.util.*;
import static java.lang.Math.*;

public class VicsekUpdate implements UpdateRule {
    @Override
    public void updateThetas(List<Agent> agents, Grid grid, double eta, Random rng, boolean includeSelf) {
        for (Agent a : agents){
            var nb = grid.neighborsOf(a, includeSelf);
            double sx = 0, sy = 0;
            for (Agent b : nb) { sx += cos(b.theta); sy += sin(b.theta); }
            if (nb.isEmpty()){ sx = cos(a.theta); sy = sin(a.theta); }
            double avg = atan2(sy, sx);
            double noise = (rng.nextDouble() - 0.5) * eta;
            a.nextTheta = avg + noise;
        }
        for (Agent a : agents) a.theta = a.nextTheta;
    }
}
