package flock;

import java.util.*;

public class VoterUpdate implements UpdateRule {
    @Override
    public void updateThetas(List<Agent> agents, Grid grid, double eta, Random rng, boolean includeSelf) {
        for (Agent a : agents){
            var nb = grid.neighborsOf(a, includeSelf);
            double base = a.theta;
            if (!nb.isEmpty()){
                Agent pick = nb.get(rng.nextInt(nb.size()));
                base = pick.theta;
            }
            double noise = (rng.nextDouble() - 0.5) * eta;
            a.nextTheta = base + noise;
        }
        for (Agent a : agents) a.theta = a.nextTheta;
    }
}
