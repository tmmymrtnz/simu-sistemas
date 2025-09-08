package simulation;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import static java.lang.Math.*;

public final class TargetEventSim {

    private enum Kind { NONE, WALL, PARTICLE }

    private static final class NextParticleEvent {
        Kind kind = Kind.NONE;
        int partner = -1;
        Wall wall = null;
        double dt = Double.POSITIVE_INFINITY;
        double bestWallDt = Double.POSITIVE_INFINITY;
        double bestPartDt = Double.POSITIVE_INFINITY;
    }

    private static final class Event implements Comparable<Event> {
        double tAbs; int i; int stamp;
        Event(double tAbs, int i, int stamp){ this.tAbs=tAbs; this.i=i; this.stamp=stamp; }
        public int compareTo(Event o){ return Double.compare(this.tAbs, o.tAbs); }
    }

    private final List<Agent> A;
    private final List<Wall>  W;
    private final int N;

    private final double Lf, L;             // para presiones/recintos
    private final PressureTracker pressure; // acumulador de presión

    private double T = 0.0;
    private final NextParticleEvent[] next;
    private final int[] planStamp;
    private final List<Set<Integer>> dependents;

    private final PriorityQueue<Event> pq = new PriorityQueue<>();
    private boolean verbose = false;

    public TargetEventSim(List<Agent> agents, List<Wall> walls, double L_fixed, double L){
        this.A = agents; this.W = walls; this.N = agents.size();
        this.Lf = L_fixed; this.L = L;
        this.pressure = new PressureTracker(L_fixed, L);
        this.next = new NextParticleEvent[N];
        this.planStamp = new int[N];
        this.dependents = new ArrayList<>(N);
        for (int i=0;i<N;i++){ next[i]=new NextParticleEvent(); dependents.add(new HashSet<>()); }
    }

    /** Corre la simulación por eventos hasta tMax, logea eventos y presión. */
    public void run(double tMax, Path eventsPath, Path pressurePath, boolean verbose) throws IOException {
        this.verbose = verbose;

        // Asegura extensión .txt
        eventsPath   = ensureTxt(eventsPath);
        pressurePath = ensureTxt(pressurePath);

        for (int i=0;i<N;i++) recomputeNextAndEnqueue(i);

        try (BufferedWriter evLog = Files.newBufferedWriter(eventsPath, StandardCharsets.UTF_8,
                                                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
             BufferedWriter pLog  = Files.newBufferedWriter(pressurePath, StandardCharsets.UTF_8,
                                                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            writeSnapshot(evLog, "INIT");
            writePressureHeader(pLog);
            writePressureSample(pLog); // t=0
            if (verbose) System.out.printf("== INIT  T=%.9f  N=%d%n", T, N);

            int steadyCount = 0;

            while (!pq.isEmpty() && T < tMax){
                Event ev = pq.poll();
                if (ev.stamp != planStamp[ev.i]) continue;
                if (ev.tAbs < T) continue;

                // --- evitar procesar eventos más allá de tMax ---
                if (ev.tAbs > tMax) {
                    trackVirtualCrossings(tMax - T); // última contribución virtual
                    advanceAll(tMax - T);
                    T = tMax;
                    writeSnapshot(evLog, "END");
                    writePressureSample(pLog);
                    if (verbose) System.out.printf("== END   T=%.9f (clamped to tMax)%n", T);
                    return;
                }

                // Antes de avanzar, registrar cruces por la pared imaginaria x=Lf (Recinto 1)
                trackVirtualCrossings(ev.tAbs - T);

                // Avanzar al evento
                double dt = ev.tAbs - T;
                if (dt > 0) {
                    advanceAll(dt);
                    if (verbose)
                        System.out.printf("[advance] Δt=%.9f  from T=%.9f -> T'=%.9f (i=%d)%n",
                                dt, T, ev.tAbs, A.get(ev.i).id);
                    T = ev.tAbs;
                }

                NextParticleEvent nx = next[ev.i];
                if (nx.kind == Kind.NONE || !Double.isFinite(nx.dt)) continue;

                if (nx.kind == Kind.WALL) {
                    Agent ai = A.get(ev.i);
                    double[] nrm = Collisions.wallImpactNormal(ai, nx.wall);

                    // impulso normal previo a la reflexión
                    double vn = ai.vx*nrm[0] + ai.vy*nrm[1];
                    double J  = (vn < 0 ? 2.0 * ai.mass * (-vn) : 0.0);
                    pressure.addRealWallImpulse(nx.wall, J);

                    if (verbose) {
                        System.out.printf("[event]  PW  T=%.9f  i=%d  wall=(%.4f,%.4f)-(%.4f,%.4f)  cause: min{ wall=%.6f , part=%.6f }=%.6f  | J=%.6e%n",
                                T, ai.id, nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2,
                                nx.bestWallDt, nx.bestPartDt, nx.dt, J);
                    }

                    Collisions.resolveParticleWall(ai, nrm[0], nrm[1]);
                    microSeparate(ai, nrm[0], nrm[1]);

                    writeEvent(evLog, String.format("PW i=%d wall=(%.6f,%.6f)-(%.6f,%.6f)",
                            ai.id, nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2));

                    // Replanificar i y watchers
                    Set<Integer> watchers = new HashSet<>(dependents.get(ev.i));
                    recomputeNextAndEnqueue(ev.i);
                    if (verbose)
                        System.out.printf("[replan]  i=%d  watchers=%d%n", ai.id, watchers.size());
                    for (int k : watchers) recomputeNextAndEnqueue(k);

                } else { // PARTICLE
                    int j = nx.partner;
                    Agent ai = A.get(ev.i), aj = A.get(j);
                    if (verbose) {
                        System.out.printf("[event]  PP  T=%.9f  i=%d  j=%d  cause: min{ wall=%.6f , part=%.6f }=%.6f%n",
                                T, ai.id, aj.id, nx.bestWallDt, nx.bestPartDt, nx.dt);
                    }
                    Collisions.resolveParticleParticle(ai, aj);

                    // Separador mínimo
                    double dx = aj.x - ai.x, dy = aj.y - ai.y, d = hypot(dx,dy);
                    if (d > Constants.TIME_EPS) {
                        double nxn = dx/d, nyn = dy/d, sep=Constants.TIME_EPS;
                        ai.x -= nxn*sep; ai.y -= nyn*sep;
                        aj.x += nxn*sep; aj.y += nyn*sep;
                    }

                    writeEvent(evLog, String.format("PP i=%d j=%d", ai.id, aj.id));

                    // Replanificar i, j y watchers
                    Set<Integer> watchersI = new HashSet<>(dependents.get(ev.i));
                    Set<Integer> watchersJ = new HashSet<>(dependents.get(j));
                    recomputeNextAndEnqueue(ev.i);
                    recomputeNextAndEnqueue(j);
                    if (verbose)
                        System.out.printf("[replan]  i=%d watchers=%d | j=%d watchers=%d%n",
                                ai.id, watchersI.size(), aj.id, watchersJ.size());
                    for (int k : watchersI) recomputeNextAndEnqueue(k);
                    for (int k : watchersJ) if (k!=ev.i) recomputeNextAndEnqueue(k);
                }

                writeSnapshot(evLog, "POST");
                writePressureSample(pLog);

                // Chequeo de “estado estacionario”: P1 ≈ P2 durante varias muestras
                double p1 = pressure.pressureLeft(T), p2 = pressure.pressureRight(T);
                double rel = abs(p1 - p2) / max(max(p1, p2), 1e-12);
                if (rel <= Constants.PRESSURE_TOL) steadyCount++;
                else steadyCount = 0;

                if (steadyCount >= Constants.STEADY_SAMPLES) {
                    if (verbose) System.out.printf("== END   T=%.9f (steady: |P1-P2|/max ≤ %.3f for %d samples)%n",
                            T, Constants.PRESSURE_TOL, Constants.STEADY_SAMPLES);
                    writeSnapshot(evLog, "END");
                    writePressureSample(pLog);
                    return;
                }

                if (T >= tMax) break;
            }

            writeSnapshot(evLog, "END");
            writePressureSample(pLog);
            if (verbose) System.out.printf("== END   T=%.9f%n", T);
        }
    }

    // ---------- avance y cruces virtuales ----------

    private void advanceAll(double dt){
        if (dt <= 0) return;
        for (Agent a: A){ a.x += a.vx*dt; a.y += a.vy*dt; }
    }

    /** Registra impulsos “virtuales” si alguna partícula del recinto izquierdo cruza x=Lf. */
    private void trackVirtualCrossings(double dt){
        if (dt <= 0) return;
        final double Lf = this.Lf;

        for (Agent a : A) {
            double x0 = a.x, y0 = a.y;
            if (!pressure.isInsideLeft(x0, y0)) continue;     // solo las que están dentro del recinto 1
            if (a.vx <= 0) continue;                          // debe ir hacia +x
            double tCross = (Lf - x0) / a.vx;                  // tiempo de cruce con la vertical x=Lf
            if (tCross > 0 && tCross <= dt) {
                double yCross = y0 + a.vy * tCross;
                if (yCross >= -1e-12 && yCross <= Lf + 1e-12) {
                    double J = 2.0 * a.mass * a.vx;           // impulso virtual |Δp_n|
                    pressure.addVirtualLeftImpulse(J);
                    if (verbose) {
                        System.out.printf("[virt]   T≈%.9f  id=%d crosses x=Lf -> J=%.6e%n",
                                (T + tCross), a.id, J);
                    }
                }
            }
        }
    }

    // ---------- planificación ----------

    private void recomputeNextAndEnqueue(int i){
        NextParticleEvent old = next[i];
        if (old.kind == Kind.PARTICLE && old.partner >= 0) {
            dependents.get(old.partner).remove(i);
        }

        Agent ai = A.get(i);

        double bestDt     = Double.POSITIVE_INFINITY;
        Kind   bestKind   = Kind.NONE;
        int    bestPartner= -1;
        Wall   bestWall   = null;

        double bestWallDt = Double.POSITIVE_INFINITY;
        double bestPartDt = Double.POSITIVE_INFINITY;

        // paredes
        for (Wall w : W){
            double t = Collisions.timeToWall(ai, w);
            if (t < bestWallDt) { bestWallDt = t; bestWall = w; }
            if (t < bestDt)     { bestDt = t; bestKind = Kind.WALL; bestWall = w; bestPartner = -1; }
        }
        // partículas
        for (int j=0;j<N;j++){
            if (j==i) continue;
            double t = Collisions.timeToParticle(ai, A.get(j));
            if (t < bestPartDt) { bestPartDt = t; }
            if (t < bestDt)     { bestDt = t; bestKind = Kind.PARTICLE; bestPartner = j; bestWall = null; }
        }

        NextParticleEvent nx = next[i];
        nx.kind = bestKind; nx.partner = bestPartner; nx.wall = bestWall; nx.dt = bestDt;
        nx.bestWallDt = bestWallDt; nx.bestPartDt = bestPartDt;

        if (nx.kind == Kind.PARTICLE && nx.partner >= 0) dependents.get(nx.partner).add(i);

        planStamp[i]++;
        if (Double.isFinite(nx.dt)) pq.add(new Event(T + nx.dt, i, planStamp[i]));

        if (verbose) {
            if (nx.kind == Kind.WALL && nx.wall != null) {
                System.out.printf("[plan]    T=%.9f  i=%d  next=WALL dt=%.9f  (bestWall=%.9f, bestPart=%.9f)  wall=(%.4f,%.4f)-(%.4f,%.4f)%n",
                        T, A.get(i).id, nx.dt, nx.bestWallDt, nx.bestPartDt,
                        nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2);
            } else if (nx.kind == Kind.PARTICLE) {
                System.out.printf("[plan]    T=%.9f  i=%d  next=PART dt=%.9f  (bestWall=%.9f, bestPart=%.9f)  j=%d%n",
                        T, A.get(i).id, nx.dt, nx.bestWallDt, nx.bestPartDt, A.get(nx.partner).id);
            } else {
                System.out.printf("[plan]    T=%.9f  i=%d  next=NONE%n", T, A.get(i).id);
            }
        }
    }

    // ---------- logging ----------

    private static Path ensureTxt(Path p){
        String base = p.getFileName().toString();
        if (!base.toLowerCase().endsWith(".txt")) {
            int dot = base.lastIndexOf('.');
            String name = (dot >= 0 ? base.substring(0, dot) : base) + ".txt";
            return p.resolveSibling(name);
        }
        return p;
    }

    private void writeSnapshot(BufferedWriter out, String tag) throws IOException {
        out.write(String.format(java.util.Locale.US, "SNAPSHOT %s t=%.9f", tag, T));
        for (Agent a : A) {
            out.write(String.format(java.util.Locale.US,
                    " | id=%d x=%.6f y=%.6f vx=%.6f vy=%.6f r=%.6f",
                    a.id, a.x, a.y, a.vx, a.vy, a.r));
        }
        out.newLine();
    }

    private void writeEvent(BufferedWriter out, String desc) throws IOException {
        out.write(String.format(java.util.Locale.US, "EVENT t=%.9f %s", T, desc));
        out.newLine();
    }

    private void writePressureHeader(BufferedWriter out) throws IOException {
        out.write(String.format(java.util.Locale.US,
                "# t  P_left  P_right   | perimLeft=%.6f perimRight=%.6f%n",
                pressure.getPerimLeft(), pressure.getPerimRight()));
    }

    private void writePressureSample(BufferedWriter out) throws IOException {
        out.write(String.format(java.util.Locale.US, "%.9f %.9e %.9e%n",
                T, pressure.pressureLeft(T), pressure.pressureRight(T)));
    }

    // ---------- helpers ----------

    private static void microSeparate(Agent a, double nx, double ny){
        double sep = Constants.TIME_EPS;
        a.x += nx*sep; a.y += ny*sep;
    }
}
