package cim;

import java.util.*;

public class CellIndexMethod {
    private final double L, rc, cellSide;
    private final int M;
    private final boolean periodic;
    private final boolean useRadii;          // umbral: rc (puntuales) o rc + ri + rj
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

        // poblar celdas
        for (List<Particle> l : cells) l.clear();
        for (Particle p : ps) cells[flat(cidx(p.x), cidx(p.y))].add(p);

        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Particle p : ps) res.put(p.id, new HashSet<>());

        // offsets del semiplano
        final int[][] OFF = { {0,0}, {1,0}, {1,1}, {0,1}, {-1,1} };

        for (int cx = 0; cx < M; cx++) {
            for (int cy = 0; cy < M; cy++) {
                List<Particle> A = cells[flat(cx, cy)];
                if (A.isEmpty()) continue;

                for (int[] d : OFF) {
                    int nx = cx + d[0], ny = cy + d[1];
                    if (periodic) {
                        nx = (nx + M) % M;
                        ny = (ny + M) % M;
                    } else {
                        if (nx < 0 || ny < 0 || nx >= M || ny >= M) continue;
                    }

                    // FIX: si por wrap caemos en la misma celda y el offset no es (0,0),
                    // salteamos para evitar comparar A con A en la rama de celdas distintas (M=1).
                    if ((nx == cx && ny == cy) && !(d[0] == 0 && d[1] == 0)) {
                        continue;
                    }

                    List<Particle> B = cells[flat(nx, ny)];
                    if (B.isEmpty()) continue;

                    if (d[0] == 0 && d[1] == 0) {
                        // misma celda: i<j
                        for (int i = 0; i < A.size(); i++) {
                            Particle a = A.get(i);
                            for (int j = i + 1; j < A.size(); j++) {
                                Particle b = A.get(j);
                                comparisons++;
                                double thr = useRadii ? (rc + a.r + b.r) : rc;
                                if (Particle.dist2(a, b, L, periodic) <= thr * thr) {
                                    res.get(a.id).add(b.id);
                                    res.get(b.id).add(a.id);
                                }
                            }
                        }
                    } else {
                        // celdas distintas (o semiplano): evitar self-pair por seguridad
                        for (Particle a : A) {
                            for (Particle b : B) {
                                if (a.id == b.id) continue; // guard extra
                                comparisons++;
                                double thr = useRadii ? (rc + a.r + b.r) : rc;
                                if (Particle.dist2(a, b, L, periodic) <= thr * thr) {
                                    res.get(a.id).add(b.id);
                                    res.get(b.id).add(a.id);
                                }
                            }
                        }
                    }
                }
            }
        }
        return res;
    }


    public Map<Integer, List<Integer>> brute_force(List<Particle> ps) {
        Map<Integer, List<Integer>> neighboursMap = new HashMap<>();
        for (int i=0; i< ps.size(); i++){
            List<Integer> neighbours = new ArrayList<>();
            for (int j=0; j< ps.size(); j++){
                if(i==j)
                    continue;
                double dist = Particle.dist2(ps.get(i), ps.get(j), L, periodic);
                double thr = useRadii ? (rc + ps.get(i).r + ps.get(j).r) : rc;
                if (thr * thr >= dist) {
                    neighbours.add(ps.get(j).id);
                }
            }
            neighboursMap.put(ps.get(i).id, neighbours);
        }
        return neighboursMap;
    }
}
