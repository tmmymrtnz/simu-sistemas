package flock;

import java.util.*;

/** Grilla de celdas para consultas de vecindad con contorno periódico.
 *  Soporta:
 *   - Puntuales (sin radio): umbral centro-centro R
 *   - Con radio: umbral borde-a-borde R + r_i + r_j
 *  Rebuild + consulta en 3x3 (neighborsOf) o pasada única con semiplano (adjacency).
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
    public final double rMax;           // para criterio geométrico del M seguro

    // Usamos lista de listas para evitar arrays genéricos
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

        // Inicializar M*M celdas vacías
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

    /** Reubica todos los agentes en celdas (¡llamar una vez por paso de simulación!). */
    public void rebuild(List<Agent> as){
        for (List<Agent> l : cells) l.clear();
        for (Agent a : as) cells.get(flat(cidx(a.x), cidx(a.y))).add(a);
    }

    /** Consulta local: vecinos de 'a' dentro del umbral mirando sólo 3x3 celdas.
     *   - Puntuales:     dist(a,b) ≤ R
     *   - Con radio:     dist(a,b) ≤ R + r_a + r_b   (borde-a-borde)
     *  includeSelf: si true, puede incluir a 'a' (útil en Vicsek).
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
                    double thr = useRadii ? (R + ra + radiusOf(b)) : R;
                    if (dist2(a, b) <= thr*thr) res.add(b);
                }
            }
        }
        return res;
    }

    /** Pasada única “CIM clásico” con semiplano (self, E, NE, N, NW).
     *  Devuelve mapa id -> lista de vecinos (listas mutuas, sin doble conteo).
     *  Más eficiente cuando necesitás la vecindad de TODOS en cada paso.
     */
    public Map<Integer, List<Agent>> adjacency(boolean includeSelf){
        Map<Integer, List<Agent>> adj = new HashMap<>();
        // pre-crear listas
        for (List<Agent> cell : cells) {
            for (Agent a : cell) adj.put(a.id, new ArrayList<>());
        }
        // offsets del semiplano
        final int[][] OFF = { {0,0}, {1,0}, {1,1}, {0,1}, {-1,1} };

        for (int cy=0; cy<M; cy++){
            for (int cx=0; cx<M; cx++){
                List<Agent> A = cells.get(flat(cx,cy));
                if (A.isEmpty()) continue;

                for (int[] d : OFF){
                    int nx = cx + d[0], ny = cy + d[1];
                    if (periodic){
                        nx = (nx + M) % M; ny = (ny + M) % M;
                    } else {
                        if (nx<0 || ny<0 || nx>=M || ny>=M) continue;
                    }
                    int fid = flat(nx, ny);

                    // Si el vecino “envuelve” a la misma celda (p.ej. M=1), evitamos duplicar:
                    if (fid == flat(cx,cy) && !(d[0]==0 && d[1]==0)) continue;

                    List<Agent> B = cells.get(fid);
                    if (B.isEmpty()) continue;

                    if (d[0]==0 && d[1]==0){
                        // pares internos de la misma celda
                        for (int i=0;i<A.size();i++){
                            Agent a = A.get(i);
                            if (includeSelf) adj.get(a.id).add(a);
                            for (int j=i+1;j<A.size();j++){
                                Agent b = A.get(j);
                                double thr = useRadii ? (R + radiusOf(a) + radiusOf(b)) : R;
                                if (dist2(a,b) <= thr*thr){
                                    adj.get(a.id).add(b);
                                    adj.get(b.id).add(a);
                                }
                            }
                        }
                    } else {
                        // pares entre A y B (semiplano)
                        for (Agent a : A){
                            double ra = useRadii ? radiusOf(a) : 0.0;
                            for (Agent b : B){
                                double thr = useRadii ? (R + ra + radiusOf(b)) : R;
                                if (dist2(a,b) <= thr*thr){
                                    adj.get(a.id).add(b);
                                    adj.get(b.id).add(a);
                                }
                            }
                        }
                    }
                }
            }
        }
        return adj;
    }
}
