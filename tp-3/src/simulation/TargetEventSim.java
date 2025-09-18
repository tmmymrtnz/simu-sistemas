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

    private final PressureTracker pressure;
    private double T = 0.0;

    private final NextParticleEvent[] next;
    private final int[] planStamp;
    private final List<Set<Integer>> dependents;
    private final PriorityQueue<Event> pq = new PriorityQueue<>();

    private boolean verbose = false;

    public TargetEventSim(List<Agent> agents, List<Wall> walls, double L_fixed, double L){
        this.A = agents; this.W = walls; this.N = agents.size();
        this.pressure = new PressureTracker(L_fixed, L);
        this.next = new NextParticleEvent[N];
        this.planStamp = new int[N];
        this.dependents = new ArrayList<>(N);
        for (int i=0;i<N;i++){ next[i]=new NextParticleEvent(); dependents.add(new HashSet<>()); }
    }

    /** Corre la simulación por eventos hasta tMax; presión POR-INTERVALO a Δt fijo. */
    public void run(double tMax, Path eventsPath, Path pressurePath, boolean verbose) throws IOException {
        this.verbose = verbose;

        final double sampleDt = Constants.PRESSURE_SAMPLE_DT;
        double nextSampleT = sampleDt; // primera marca después de t=0

        eventsPath   = ensureTxt(eventsPath);
        pressurePath = ensureTxt(pressurePath);

        for (int i=0;i<N;i++) recomputeNextAndEnqueue(i);

        try (BufferedWriter evLog = Files.newBufferedWriter(eventsPath, StandardCharsets.UTF_8,
                                                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
             BufferedWriter pLog  = Files.newBufferedWriter(pressurePath, StandardCharsets.UTF_8,
                                                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            writeSnapshot(evLog, "INIT");
            writePressureHeader(pLog, sampleDt);
            // Fila inicial en t=0 (intervalo “vacío”)
            sampleAndLogInterval(pLog);

            if (verbose) System.out.printf("== INIT  T=%.9f  N=%d%n", T, N);

            int steadyCount = 0;

            while (!pq.isEmpty() && T < tMax){
                Event ev = pq.poll();
                if (ev.stamp != planStamp[ev.i]) continue;
                if (ev.tAbs < T) continue;

                // 1) Muestrear marcas estrictamente antes del próximo evento
                while (nextSampleT < ev.tAbs && nextSampleT <= tMax) {
                    advanceAll(nextSampleT - T);
                    T = nextSampleT;
                    double[] pi = sampleAndLogInterval(pLog);
                    steadyCount = updateSteadyCount(steadyCount, pi[0], pi[1]);
                    nextSampleT += sampleDt;
                }

                // 2) Si el evento cae más allá de tMax, avanzar y cerrar
                if (ev.tAbs > tMax) {
                    while (nextSampleT <= tMax) {
                        advanceAll(nextSampleT - T);
                        T = nextSampleT;
                        double[] pi = sampleAndLogInterval(pLog);
                        steadyCount = updateSteadyCount(steadyCount, pi[0], pi[1]);
                        nextSampleT += sampleDt;
                    }
                    if (T < tMax) {
                        advanceAll(tMax - T);
                        T = tMax;
                        writeSnapshot(evLog, "END");
                        double[] pi = sampleAndLogInterval(pLog);
                        updateSteadyCount(steadyCount, pi[0], pi[1]);
                    } else {
                        writeSnapshot(evLog, "END");
                    }
                    if (verbose) System.out.printf("== END   T=%.9f (clamped to tMax)%n", T);
                    return;
                }

                // 3) Avanzar exactamente al evento y procesarlo
                if (ev.tAbs > T) {
                    advanceAll(ev.tAbs - T);
                    T = ev.tAbs;
                }

                NextParticleEvent nx = next[ev.i];
                if (nx.kind == Kind.NONE || !Double.isFinite(nx.dt)) continue;

                if (nx.kind == Kind.WALL) {
                    Agent ai = A.get(ev.i);

                    // --- impulso antes de modificar velocidad ---
                    double[] nrm1 = Collisions.wallImpactNormal(ai, nx.wall);
                    double vn1 = ai.vx*nrm1[0] + ai.vy*nrm1[1];
                    double J  = (vn1 < 0 ? 2.0 * ai.mass * (-vn1) : 0.0);

                    // ¿Contacto en vértice? (reparte impulso y doble reflexión)
                    final double VERT_S_EPS = 1e-9;
                    boolean isVertex = false;
                    Wall w1 = nx.wall;
                    Wall w2 = null;

                    double ux = w1.x2 - w1.x1, uy = w1.y2 - w1.y1;
                    double L2 = ux*ux + uy*uy;
                    if (L2 > 0) {
                        double wx = ai.x - w1.x1, wy = ai.y - w1.y1;
                        double s = (wx*ux + wy*uy)/L2; // proyección (no clamp)
                        if (s <= VERT_S_EPS || s >= 1.0 - VERT_S_EPS) {
                            double vx = (s <= VERT_S_EPS) ? w1.x1 : w1.x2;
                            double vy = (s <= VERT_S_EPS) ? w1.y1 : w1.y2;
                            w2 = findAdjacentWallAtVertex(w1, vx, vy);
                            isVertex = (w2 != null);
                        }
                    }

                    if (isVertex) {
                        double[] nrm2 = Collisions.wallImpactNormal(ai, w2);
                        double wght1 = Math.abs(ai.vx*nrm1[0] + ai.vy*nrm1[1]);
                        double wght2 = Math.abs(ai.vx*nrm2[0] + ai.vy*nrm2[1]);
                        if (wght1 + wght2 < 1e-12) { wght1 = wght2 = 1.0; }

                        double J1 = J * (wght1 / (wght1 + wght2));
                        double J2 = J - J1;

                        pressure.addRealWallImpulse(w1, J1);
                        pressure.addRealWallImpulse(w2, J2);

                        Collisions.resolveParticleWall(ai, nrm1[0], nrm1[1]);
                        Collisions.resolveParticleWall(ai, nrm2[0], nrm2[1]);

                        double bx = nrm1[0] + nrm2[0], by = nrm1[1] + nrm2[1];
                        double bl = hypot(bx, by);
                        if (bl < 1e-12) { bx = nrm1[0]; by = nrm1[1]; bl = hypot(bx, by); }
                        microSeparate(ai, bx/bl, by/bl);

                        writeEvent(evLog, String.format(
                            "PW-VERTEX i=%d w1=(%.6f,%.6f)-(%.6f,%.6f) w2=(%.6f,%.6f)-(%.6f,%.6f)",
                            ai.id, w1.x1, w1.y1, w1.x2, w1.y2, w2.x1, w2.y1, w2.x2, w2.y2));

                    } else {
                        pressure.addRealWallImpulse(w1, J);
                        Collisions.resolveParticleWall(ai, nrm1[0], nrm1[1]);
                        microSeparate(ai, nrm1[0], nrm1[1]);

                        writeEvent(evLog, String.format("PW i=%d wall=(%.6f,%.6f)-(%.6f,%.6f)",
                                ai.id, w1.x1, w1.y1, w1.x2, w1.y2));
                    }

                    Set<Integer> watchers = new HashSet<>(dependents.get(ev.i));
                    recomputeNextAndEnqueue(ev.i);
                    for (int k : watchers) recomputeNextAndEnqueue(k);

                } else { // PARTICLE
                    int j = nx.partner;
                    Agent ai = A.get(ev.i), aj = A.get(j);

                    Collisions.resolveParticleParticle(ai, aj);

                    double dx = aj.x - ai.x, dy = aj.y - ai.y, d = hypot(dx,dy);
                    if (d > Constants.TIME_EPS) {
                        double nxn = dx/d, nyn = dy/d, sep=Constants.TIME_EPS;
                        ai.x -= nxn*sep; ai.y -= nyn*sep;
                        aj.x += nxn*sep; aj.y += nyn*sep;
                    }

                    writeEvent(evLog, String.format("PP i=%d j=%d", ai.id, aj.id));

                    Set<Integer> watchersI = new HashSet<>(dependents.get(ev.i));
                    Set<Integer> watchersJ = new HashSet<>(dependents.get(j));
                    recomputeNextAndEnqueue(ev.i);
                    recomputeNextAndEnqueue(j);
                    for (int k : watchersI) recomputeNextAndEnqueue(k);
                    for (int k : watchersJ) if (k!=ev.i) recomputeNextAndEnqueue(k);
                }

                // 4) Snapshot post-evento
                writeSnapshot(evLog, "POST");

                // 5) Si hay marca EXACTA en T (la del segundo), muestrear después del evento
                if (abs(nextSampleT - T) <= 1e-12) {
                    double[] pi = sampleAndLogInterval(pLog);
                    steadyCount = updateSteadyCount(steadyCount, pi[0], pi[1]);
                    nextSampleT += sampleDt;
                }

                // 6) ¿steady?
                if (steadyCount >= Constants.STEADY_SAMPLES) {
                    writeSnapshot(evLog, "END");
                    double[] pi = sampleAndLogInterval(pLog);
                    updateSteadyCount(steadyCount, pi[0], pi[1]);
                    if (verbose) System.out.printf(
                        "== END   T=%.9f (steady por-intervalo: |P1-P2|/mean ≤ %.3f por %d muestras)%n",
                        T, Constants.PRESSURE_TOL, Constants.STEADY_SAMPLES);
                    return;
                }
            }

            // Si se vació la PQ antes de tMax: completar marcas y cerrar
            while (T < tMax) {
                double upTo = Math.min(T + Constants.PRESSURE_SAMPLE_DT, tMax);
                advanceAll(upTo - T);
                T = upTo;
                double[] pi = sampleAndLogInterval(pLog);
                updateSteadyCount(0, pi[0], pi[1]);
            }
            writeSnapshot(evLog, "END");
            if (verbose) System.out.printf("== END   T=%.9f%n", T);
        }
    }

    // ---------- steady helper ----------
    private int updateSteadyCount(int steadyCount, double p1, double p2){
        double meanP = 0.5 * (p1 + p2);
        if (meanP > Constants.MIN_MEAN_PRESSURE) {
            double rel = Math.abs(p1 - p2) / meanP;
            if (rel <= Constants.PRESSURE_TOL) return steadyCount + 1;
        }
        return 0;
    }

    // ---------- avance ----------
    private void advanceAll(double dt){
        if (dt <= 0) return;
        for (Agent a: A){ a.x += a.vx*dt; a.y += a.vy*dt; }
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

        for (Wall w : W){
            double t = Collisions.timeToWall(ai, w);
            if (t < bestWallDt) { bestWallDt = t; bestWall = w; }
            if (t < bestDt)     { bestDt = t; bestKind = Kind.WALL; bestWall = w; bestPartner = -1; }
        }
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
    }

    // ---------- logging helpers ----------
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

    private void writePressureHeader(BufferedWriter out, double sampleDt) throws IOException {
        out.write(String.format(java.util.Locale.US,
                "# t  P_left  P_right   (INTERVAL, dt=%.6f s) | perimLeft=%.6f perimRight=%.6f%n",
                sampleDt, pressure.getPerimLeft(), pressure.getPerimRight()));
    }

    /** Mide presión POR-INTERVALO (último Δt) y la escribe en el log. */
    private double[] sampleAndLogInterval(BufferedWriter out) throws IOException {
        double[] pi = pressure.sampleInterval(T); // [P_left_dt, P_right_dt]
        if (T > 0 && pi[0] == 0.0 && pi[1] == 0.0) return pi;
        double pTotal = (pi[0] * pressure.getPerimLeft() + pi[1] * pressure.getPerimRight()) /
                        (pressure.getPerimLeft() + pressure.getPerimRight());
        out.write(String.format(java.util.Locale.US, "%.9f %.9e %.9e %.9e%n", T, pi[0], pi[1], pTotal));
        return pi;
    }

    private static void microSeparate(Agent a, double nx, double ny){
        double sep = Constants.TIME_EPS;
        a.x += nx*sep; a.y += ny*sep;
    }


    private void separateParticles(Agent a, Agent b, double dx, double dy, double d) {
        final double MIN_DISTANCE = a.r + b.r;
        final double OVERLAP_THRESHOLD = 0.001 * MIN_DISTANCE;
      if (d == 0) {
           a.x += (Math.random() - 0.5) * 1e-6;
            a.y += (Math.random() - 0.5) * 1e-6;
            b.x += (Math.random() - 0.5) * 1e-6;
            b.y += (Math.random() - 0.5) * 1e-6;
        } else if (d < MIN_DISTANCE + OVERLAP_THRESHOLD) {
            double sep = Constants.TIME_EPS;
           a.x -= (dx / d) * sep; a.y -= (dy / d) * sep;
            b.x += (dx / d) * sep; b.y += (dy / d) * sep;
        }
    }

    // ---------- util vértices ----------
    private static boolean near(double a, double b, double eps){ return Math.abs(a-b) <= eps; }

   
    /** Busca una pared contigua que comparta el vértice (vx,vy) con w (excluye w). */
    private Wall findAdjacentWallAtVertex(Wall w, double vx, double vy){
        final double EPSV = 1e-9;
        for (Wall c : W){
            if (c == w) continue;
            boolean share =
                (near(c.x1, vx, EPSV) && near(c.y1, vy, EPSV)) ||
                (near(c.x2, vx, EPSV) && near(c.y2, vy, EPSV));
            if (share) return c;
        }
        return null;
    }

}
