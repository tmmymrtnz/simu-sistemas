package tp5.simulacion.cim;

import tp5.simulacion.core.Agent;
import java.util.*;

/**
 * Adaptation of the TP1 Cell Index Method to work with the pedestrian simulation agents.
 * Uses a square grid partition and computes neighbours within the interaction radius.
 */
public class CellIndexMethod {
    private final double L;
    private final double rc;
    private final int M;
    private final boolean periodic;
    private final boolean useRadii;
    private final double cellSide;
    private final List<Agent>[] cells;

    public long comparisons = 0;

    @SuppressWarnings("unchecked")
    public CellIndexMethod(double L, double rc, int M, boolean periodic, boolean useRadii) {
        this.L = L;
        this.rc = rc;
        this.M = M;
        this.periodic = periodic;
        this.useRadii = useRadii;
        this.cellSide = L / M;
        this.cells = new ArrayList[M * M];
        for (int i = 0; i < cells.length; i++) {
            cells[i] = new ArrayList<>();
        }
    }

    private int cidx(double c) {
        int idx = (int) Math.floor(c / cellSide);
        return Math.min(Math.max(idx, 0), M - 1);
    }

    private int flat(int cx, int cy) {
        return cy * M + cx;
    }

    public Map<Integer, Set<Integer>> neighbours(List<Agent> agents) {
        comparisons = 0;
        for (List<Agent> cell : cells) {
            cell.clear();
        }
        for (Agent agent : agents) {
            int cx = cidx(agent.getPosition().x());
            int cy = cidx(agent.getPosition().y());
            cells[flat(cx, cy)].add(agent);
        }
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Agent agent : agents) {
            res.put(agent.getId(), new HashSet<>());
        }
        final int[][] OFF = {{0,0}, {1,0}, {1,1}, {0,1}, {-1,1}};
        for (int cx = 0; cx < M; cx++) {
            for (int cy = 0; cy < M; cy++) {
                List<Agent> A = cells[flat(cx, cy)];
                if (A.isEmpty()) continue;
                for (int[] d : OFF) {
                    int nx = cx + d[0], ny = cy + d[1];
                    if (periodic) {
                        nx = (nx + M) % M;
                        ny = (ny + M) % M;
                    } else {
                        if (nx < 0 || ny < 0 || nx >= M || ny >= M) continue;
                    }
                    if ((nx == cx && ny == cy) && !(d[0] == 0 && d[1] == 0)) {
                        continue;
                    }
                    List<Agent> B = cells[flat(nx, ny)];
                    if (B.isEmpty()) continue;
                    if (d[0] == 0 && d[1] == 0) {
                        for (int i = 0; i < A.size(); i++) {
                            Agent a = A.get(i);
                            for (int j = i + 1; j < A.size(); j++) {
                                Agent b = A.get(j);
                                checkPair(res, a, b);
                            }
                        }
                    } else {
                        for (Agent a : A) {
                            for (Agent b : B) {
                                if (a.getId() == b.getId()) continue;
                                checkPair(res, a, b);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    private void checkPair(Map<Integer, Set<Integer>> res, Agent a, Agent b) {
        comparisons++;
        double thr = useRadii ? (rc + a.getRadius() + b.getRadius()) : rc;
        if (distSq(a, b) <= thr * thr) {
            res.get(a.getId()).add(b.getId());
            res.get(b.getId()).add(a.getId());
        }
    }

    private double distSq(Agent a, Agent b) {
        double dx = Math.abs(a.getPosition().x() - b.getPosition().x());
        double dy = Math.abs(a.getPosition().y() - b.getPosition().y());
        if (periodic) {
            if (dx > 0.5 * L) dx = L - dx;
            if (dy > 0.5 * L) dy = L - dy;
        }
        return dx * dx + dy * dy;
    }

    public Map<Integer, List<Integer>> bruteForce(List<Agent> agents) {
        Map<Integer, List<Integer>> neighbours = new HashMap<>();
        for (Agent a : agents) {
            List<Integer> ids = new ArrayList<>();
            for (Agent b : agents) {
                if (a.getId() == b.getId()) continue;
                double thr = useRadii ? (rc + a.getRadius() + b.getRadius()) : rc;
                if (distSq(a, b) <= thr * thr) {
                    ids.add(b.getId());
                }
            }
            neighbours.put(a.getId(), ids);
        }
        return neighbours;
    }
}
