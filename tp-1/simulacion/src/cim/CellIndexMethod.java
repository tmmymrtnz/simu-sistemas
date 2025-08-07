package cim;

import java.util.*;

public class CellIndexMethod {

    private final double L, rc;
    private final int M;
    private final boolean periodic;
    private final double cellSide;
    @SuppressWarnings("unchecked")
    private final List<Particle>[] cells;   // asignada en ctor

    @SuppressWarnings("unchecked")
    public CellIndexMethod(double L, double rc, int M, boolean periodic) {
        this.L = L;
        this.rc = rc;
        this.M = M;
        this.periodic = periodic;
        this.cellSide = L / M;

        this.cells = new ArrayList[M * M];
        for (int i = 0; i < M * M; i++) cells[i] = new ArrayList<>();
    }

    private int cellIndex(double c) {
        int idx = (int) Math.floor(c / cellSide);
        return Math.min(Math.max(idx, 0), M - 1);
    }
    private int flat(int cx, int cy) { return cy * M + cx; }

    /** celdas vecinas (incluye la misma) */
    private List<int[]> neighCells(int cx, int cy) {
        List<int[]> list = new ArrayList<>(9);
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int nx = cx + dx, ny = cy + dy;
                if (periodic) {
                    nx = (nx + M) % M;
                    ny = (ny + M) % M;
                } else if (nx < 0 || ny < 0 || nx >= M || ny >= M)
                    continue;
                list.add(new int[]{nx, ny});
            }
        return list;
    }

    /** devuelve mapa id → conjunto de ids vecinos */
    public Map<Integer, Set<Integer>> neighbours(List<Particle> ps) {
        for (List<Particle> l : cells) l.clear();
        for (Particle p : ps)
            cells[flat(cellIndex(p.x), cellIndex(p.y))].add(p);

        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (Particle p : ps) res.put(p.id, new HashSet<>());

        for (int cx = 0; cx < M; cx++)
            for (int cy = 0; cy < M; cy++) {
                List<Particle> bucket = cells[flat(cx, cy)];
                if (bucket.isEmpty()) continue;
                for (int[] nb : neighCells(cx, cy)) {
                    List<Particle> other = cells[flat(nb[0], nb[1])];
                    if (other.isEmpty()) continue;
                    for (Particle a : bucket)
                        for (Particle b : other) {
                            if (a.id >= b.id) continue; // evita duplicados
                            double thresh = rc + a.r + b.r;
                            if (a.dist2(b, L, periodic) <= thresh * thresh) {
                                res.get(a.id).add(b.id);
                                res.get(b.id).add(a.id);
                            }
                        }
                }
            }
        return res;
    }
}
