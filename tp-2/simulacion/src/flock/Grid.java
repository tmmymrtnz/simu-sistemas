package flock;

import java.util.*;

/** Grilla de celdas para consultas de vecindad (Moore 3x3) con contorno periódico.
 *  Soporta:
 *   - Puntuales (sin radio): umbral centro-centro R
 *   - Con radio: umbral borde-a-borde R + r_i + r_j
 */
public class Grid {

    /** Implementar en Agent si querés trabajar con radios > 0. */
    public static interface RadiusProvider {
        double radius();
    }

    public final double L, R, cellSide;
    public final int M;                 // celdas por lado
    public final boolean periodic;
    public final boolean useRadii;      // true => borde-a-borde (R + ri + rj)
    public final double rMax;           // usado para criterio geométrico del M seguro

    // En vez de ArrayList<Agent>[] usamos una lista de listas para evitar el warning de arrays genéricos
    private final List<List<Agent>> cells;

    /** Constructor para puntuales (sin radio). Usa criterio ℓ=L/M ≥ R. */
    public Grid(double L, double R, Integer Mopt, boolean periodic){
        this(L, R, Mopt, periodic, /*useRadii=*/false, /*rMax=*/0.0);
    }

    /** Constructor general.
     *  Si useRadii=true, usa criterio geométrico ℓ=L/M ≥ (R + 2*rMax).
     *  Si Mopt es null, se elige el máximo M seguro según el criterio.
     */
    public Grid(double L, double R, Integer Mopt, boolean periodic, boolean useRadii, double rMax){
        this.L = L; this.R = R; this.periodic = periodic;
        this.useRadii = useRadii; this.rMax = Math.max(0.0, rMax);

        double minCell = useRadii ? (R + 2*this.rMax) : R;
        int Mmax = Math.max(1, (int)Math.floor(L / Math.max(1e-12, minCell)));
        this.M = (Mopt == null) ? Mmax : Math.max(1, Math.min(Mopt, Mmax));

        this.cellSide = L / this.M;

        // Inicializar M*M celdas vacías sin arrays genéricos
        this.cells = new ArrayList<>(M*M);
        for (int i=0; i<M*M; i++) this.cells.add(new ArrayList<>());
    }

    private int cidx(double c){
        int idx = (int)Math.floor(c / cellSide);
        if (idx < 0) idx = 0; if (idx >= M) idx = M - 1;
        return idx;
    }
    private int flat(int cx, int cy){ return cy*M + cx; }

    private double dist2(Agent a, Agent b){
        double dx = a.x - b.x, dy = a.y - b.y;
        if (periodic){
            if (dx > 0.5*L) dx -= L; else if (dx < -0.5*L) dx += L;
            if (dy > 0.5*L) dy -= L; else if (dy < -0.5*L) dy += L;
        }
        return dx*dx + dy*dy;
    }

    /** Devuelve el radio del agente si implementa RadiusProvider; en caso contrario 0. */
    private double radiusOf(Agent a){
        if (a instanceof RadiusProvider) return ((RadiusProvider)a).radius();
        return 0.0;
    }

    /** Reubica todos los agentes en celdas. */
    public void rebuild(List<Agent> as){
        for (List<Agent> l : cells) l.clear();
        for (Agent a : as) cells.get(flat(cidx(a.x), cidx(a.y))).add(a);
    }

    /** Vecinos de 'a' dentro del umbral:
     *   - Puntuales:     dist(a,b) ≤ R
     *   - Con radio:     dist(a,b) ≤ R + r_a + r_b   (borde-a-borde)
     *  Recorre 9 celdas (Moore). includeSelf: si true, puede incluir a 'a' en la lista (Vicsek).
     */
    public List<Agent> neighborsOf(Agent a, boolean includeSelf){
        int cx = cidx(a.x), cy = cidx(a.y);
        ArrayList<Agent> res = new ArrayList<>(16);

        // Evita duplicar celdas cuando M=1 y la envolvente periódica colapsa offsets
        HashSet<Integer> visited = new HashSet<>(9);

        final double ra = useRadii ? radiusOf(a) : 0.0;

        for (int dy=-1; dy<=1; dy++){
            for (int dx=-1; dx<=1; dx++){
                int nx = cx + dx, ny = cy + dy;
                if (periodic){
                    nx = (nx + M) % M; ny = (ny + M) % M;
                } else {
                    if (nx<0 || ny<0 || nx>=M || ny>=M) continue;
                }
                int fid = flat(nx, ny);
                if (!visited.add(fid)) continue; // celda ya procesada

                List<Agent> B = cells.get(fid);
                if (B.isEmpty()) continue;

                for (Agent b : B){
                    if (!includeSelf && a.id == b.id) continue;

                    double thr = useRadii ? (R + ra + radiusOf(b)) : R; // borde-a-borde o puntual
                    if (dist2(a, b) <= thr*thr) res.add(b);
                }
            }
        }
        return res;
    }
}
