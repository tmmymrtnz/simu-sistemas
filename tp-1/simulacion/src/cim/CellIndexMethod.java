package cim;

import java.util.*;

public class CellIndexMethod {
    private final double L, rc, cellSide;
    private final int M;
    private final boolean periodic;
    private final List<Particle>[] cells;

    public long comparisons = 0; // contador de comparaciones de pares

    @SuppressWarnings("unchecked")
    public CellIndexMethod(double L, double rc, int M, boolean periodic) {
        this.L = L; this.rc = rc; this.M = M; this.periodic = periodic;
        this.cellSide = L / M;
        this.cells = new ArrayList[M * M];
        for (int i = 0; i < cells.length; i++) cells[i] = new ArrayList<>();
    }

    private int cidx(double c) {
        int idx = (int) Math.floor(c / cellSide);
        return Math.min(Math.max(idx, 0), M - 1);
    }
    private int flat(int cx, int cy) { return cy * M + cx; }

    private List<int[]> neighCells(int cx, int cy) {
        List<int[]> list = new ArrayList<>(9);
        for (int dx=-1; dx<=1; dx++)
            for (int dy=-1; dy<=1; dy++) {
                int nx = cx + dx, ny = cy + dy;
                if (periodic) {
                    nx = (nx + M) % M; ny = (ny + M) % M;
                } else if (nx<0 || ny<0 || nx>=M || ny>=M) continue;
                list.add(new int[]{nx,ny});
            }
        return list;
    }

    /** Vecinos: mapa id -> set de ids (usa borde-a-borde: rc + ri + rj) */
    public Map<Integer, Set<Integer>> neighbours(List<Particle> ps) {
        comparisons = 0;
        for (List<Particle> l : cells) l.clear();
        for (Particle p : ps) cells[flat(cidx(p.x), cidx(p.y))].add(p);

        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Particle p : ps) res.put(p.id, new HashSet<>());

        for (int cx=0; cx<M; cx++)
            for (int cy=0; cy<M; cy++) {
                List<Particle> A = cells[flat(cx,cy)];
                if (A.isEmpty()) continue;
                for (int[] nb : neighCells(cx,cy)) {
                    List<Particle> B = cells[flat(nb[0], nb[1])];
                    if (B.isEmpty()) continue;

                    for (Particle a : A)
                        for (Particle b : B) {
                            if (a.id >= b.id) continue; // evita doble conteo
                            comparisons++;
                            double thr = rc + a.r + b.r;
                            if (Particle.dist2(a,b,L,periodic) <= thr*thr) {
                                res.get(a.id).add(b.id);
                                res.get(b.id).add(a.id);
                            }
                        }
                }
            }
        return res;
    }

    /** Fuerza bruta: O(N^2), devuelve vecinos y comparaciones usadas */
    public static class BruteOut {
        public final Map<Integer, Set<Integer>> neigh;
        public final long comparisons;
        public BruteOut(Map<Integer, Set<Integer>> n, long c){neigh=n; comparisons=c;}
    }
    public static BruteOut bruteForce(List<Particle> ps, double L, double rc, boolean periodic) {
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Particle p : ps) res.put(p.id, new HashSet<>());
        long comps = 0;
        int N = ps.size();
        for (int i=0;i<N;i++)
            for (int j=i+1;j<N;j++){
                Particle a=ps.get(i), b=ps.get(j);
                comps++;
                double thr = rc + a.r + b.r;
                if (Particle.dist2(a,b,L,periodic) <= thr*thr) {
                    res.get(a.id).add(b.id);
                    res.get(b.id).add(a.id);
                }
            }
        return new BruteOut(res, comps);
    }
}
