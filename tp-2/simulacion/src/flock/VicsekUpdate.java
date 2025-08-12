package flock;

import java.util.*;
import static java.lang.Math.*;

public class VicsekUpdate implements UpdateRule {

    private static double wrap(double th){
        double twoPi = 2*Math.PI;
        th %= twoPi;
        if (th < 0) th += twoPi;
        return th;
    }

    @Override
    public void updateThetas(List<Agent> agents, Grid grid, double eta, Random rnd, boolean includeSelf) {
        // Una sola pasada CIM (semiplano): id -> vecinos
        Map<Integer, List<Agent>> adj = grid.adjacency(includeSelf);

        // Calcular nextTheta a partir del promedio vectorial + ruido
        for (Agent a : agents) {
            List<Agent> nb = adj.get(a.id);

            double sx = 0.0, sy = 0.0;
            if (nb != null && !nb.isEmpty()) {
                for (Agent b : nb) { sx += cos(b.theta); sy += sin(b.theta); }
                double ang = atan2(sy, sx);
                double noise = (rnd.nextDouble() - 0.5) * eta;
                a.nextTheta = wrap(ang + noise);
            } else {
                // sin vecinos: mantener dirección + ruido
                double noise = (rnd.nextDouble() - 0.5) * eta;
                a.nextTheta = wrap(a.theta + noise);
            }
        }

        // Commit
        for (Agent a : agents) a.theta = a.nextTheta;
    }
}
