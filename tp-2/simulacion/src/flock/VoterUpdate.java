package flock;

import java.util.*;
import static java.lang.Math.*;

public class VoterUpdate implements UpdateRule {

    private static double wrap(double th){
        double twoPi = 2*Math.PI;
        th %= twoPi;
        if (th < 0) th += twoPi;
        return th;
    }

    @Override
    public void updateThetas(List<Agent> agents, Grid grid, double eta, Random rnd, boolean includeSelfFromProps) {
        // En voter es mejor NO incluirse a sí mismo para no "copiarse" solo.
        Map<Integer, List<Agent>> adj = grid.adjacency(/*includeSelf=*/false);

        for (Agent a : agents) {
            List<Agent> nb = adj.get(a.id);
            double noise = (rnd.nextDouble() - 0.5) * eta;

            if (nb != null && !nb.isEmpty()) {
                Agent pick = nb.get(rnd.nextInt(nb.size()));
                a.nextTheta = wrap(pick.theta + noise);
            } else {
                // sin vecinos: mantener dirección + ruido
                a.nextTheta = wrap(a.theta + noise);
            }
        }

        // Commit
        for (Agent a : agents) a.theta = a.nextTheta;
    }
}
