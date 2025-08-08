package cim;

import java.util.*;

public class CellIndexMethod {
    private final double L, rc, cellSide;
    private final int M;
    private final boolean periodic;
    private final boolean useRadii;          // <- NUEVO: cambia el umbral
    private final List<Particle>[] cells;

    public long comparisons = 0;

    @SuppressWarnings("unchecked")
    public CellIndexMethod(double L, double rc, int M, boolean periodic, boolean useRadii) {
        this.L = L; this.rc = rc; this.M = M; this.periodic = periodic; this.useRadii = useRadii;
        this.cellSide = L / M;
        this.cells = new ArrayList[M * M];
        for (int i = 0; i < cells.length; i++) cells[i] = new ArrayList<>();
    }

    private int cidx(double c) {
        int idx = (int) Math.floor(c / cellSide);
        return Math.min(Math.max(idx, 0), M - 1);
    }
    private int flat(int cx, int cy) { return cy * M + cx; }

    /** Vecinos con semiplano: self, E, NE, N, NW. */
    public Map<Integer, Set<Integer>> neighbours(List<Particle> ps) {
        comparisons = 0;
        for (List<Particle> l : cells) l.clear();
        for (Particle p : ps) cells[flat(cidx(p.x), cidx(p.y))].add(p);

        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Particle p : ps) res.put(p.id, new HashSet<>());

        final int[][] OFF = { {0,0}, {1,0}, {1,1}, {0,1}, {-1,1} };

        for (int cx=0; cx<M; cx++) {
            for (int cy=0; cy<M; cy++) {
                List<Particle> A = cells[flat(cx,cy)];
                if (A.isEmpty()) continue;

                for (int[] d : OFF) {
                    int nx = cx + d[0], ny = cy + d[1];
                    if (periodic) { nx = (nx + M) % M; ny = (ny + M) % M; }
                    else { if (nx<0 || ny<0 || nx>=M || ny>=M) continue; }

                    List<Particle> B = cells[flat(nx, ny)];
                    if (B.isEmpty()) continue;

                    if (d[0]==0 && d[1]==0) {
                        for (int i=0;i<A.size();i++) {
                            Particle a = A.get(i);
                            for (int j=i+1;j<A.size();j++) {
                                Particle b = A.get(j);
                                comparisons++;
                                double thr = useRadii ? (rc + a.r + b.r) : rc;
                                if (Particle.dist2(a,b,L,periodic) <= thr*thr) {
                                    res.get(a.id).add(b.id);
                                    res.get(b.id).add(a.id);
                                }
                            }
                        }
                    } else {
                        for (Particle a : A) for (Particle b : B) {
                            comparisons++;
                            double thr = useRadii ? (rc + a.r + b.r) : rc;
                            if (Particle.dist2(a,b,L,periodic) <= thr*thr) {
                                res.get(a.id).add(b.id);
                                res.get(b.id).add(a.id);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }
}
