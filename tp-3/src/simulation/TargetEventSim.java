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
        double dt = Double.POSITIVE_INFINITY; // tiempo relativo desde T
        // para explicar por qué: mejores candidatos parciales
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
    private final int N; //size de los agents

    private double T = 0.0;            // tiempo global
    private final NextParticleEvent[] next;         // próximo objetivo por partícula
    private final int[] planStamp;     // invalida eventos viejos de i
    private final int[] stateStamp;    // si cambia estado de i (no lo usamos aquí pero queda útil)
    private final List<Set<Integer>> dependents; // quiénes tenían a j como objetivo

    private final PriorityQueue<Event> pq = new PriorityQueue<>();

    // verbose flag (prints a stdout)
    private boolean verbose = false;

    public TargetEventSim(List<Agent> agents, List<Wall> walls){
        this.A = agents; this.W = walls; this.N = agents.size();
        this.next = new NextParticleEvent[N];
        this.planStamp = new int[N];
        this.stateStamp = new int[N];
        this.dependents = new ArrayList<>(N);
        for (int i=0;i<N;i++){ next[i]=new NextParticleEvent(); dependents.add(new HashSet<>()); }
    }

    /** Corre la simulación por eventos hasta tMax y escribe log txt. Si verbose=true, imprime a consola. */
    public void run(double tMax, Path logPath, boolean verbose) throws IOException {
        this.verbose = verbose;

        // inicial: planificar el "siguiente choque" para todos
        for (int i=0;i<N;i++) recomputeNextAndEnqueue(i);

        try (BufferedWriter out = Files.newBufferedWriter(logPath, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writeSnapshot(out, "INIT");
            if (this.verbose) System.out.printf("== INIT  T=%.9f  N=%d\n", T, N);

            while (!pq.isEmpty() && T < tMax){
                Event ev = pq.poll();
                if (ev.stamp != planStamp[ev.i]) continue; // si prendi el flag de invalido lo paso de largo  (borrado logico)
                if (ev.tAbs < T) continue;  // si es viejo, lo ignoro (tal vez no hace falta)

                double dt = ev.tAbs - T;
                if (dt > 0) {  // avanzo todo al proximo evento
                    for (Agent a: A){ a.x += a.vx*dt; a.y += a.vy*dt; }
                    if (this.verbose)
                        System.out.printf("[advance] Δt=%.9f  from T=%.9f -> T'=%.9f (i=%d)\n",
                                dt, T, ev.tAbs, A.get(ev.i).id);
                    T = ev.tAbs;
                }

                NextParticleEvent nx = next[ev.i]; //agarro el proximo evento planeado
                if (nx.kind == Kind.NONE || !Double.isFinite(nx.dt)) continue;

                if (nx.kind == Kind.WALL) {
                    Agent ai = A.get(ev.i);
                    double[] nrm = Collisions.wallImpactNormal(ai, nx.wall);
                    if (this.verbose) {
                        System.out.printf("[event]  PW  T=%.9f  i=%d  wall=(%.4f,%.4f)-(%.4f,%.4f)  cause: min{ wall=%.6f , part=%.6f }=%.6f\n",
                                T, ai.id, nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2,
                                nx.bestWallDt, nx.bestPartDt, nx.dt);
                    }
                    Collisions.resolveParticleWall(ai, nrm[0], nrm[1]);
                    microSeparate(ai, nrm[0], nrm[1]);
                    stateStamp[ev.i]++;

                    writeEvent(out, String.format("PW i=%d wall=(%.6f,%.6f)-(%.6f,%.6f)",
                            ai.id, nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2));

                    // replanificar i + watchers(i)
                    Set<Integer> watchers = new HashSet<>(dependents.get(ev.i));
                    recomputeNextAndEnqueue(ev.i); // para la partícula que disparó el evento
                    if (this.verbose)
                        System.out.printf("[replan]  i=%d  watchers=%d\n", ai.id, watchers.size());
                    for (int k : watchers) recomputeNextAndEnqueue(k); //para todas las particulas que iban a chocar conmigo

                } else { // PARTICLE
                    int j = nx.partner;
                    Agent ai = A.get(ev.i), aj = A.get(j);
                    if (this.verbose) {
                        System.out.printf("[event]  PP  T=%.9f  i=%d  j=%d  cause: min{ wall=%.6f , part=%.6f }=%.6f\n",
                                T, ai.id, aj.id, nx.bestWallDt, nx.bestPartDt, nx.dt);
                    }
                    Collisions.resolveParticleParticle(ai, aj);

                    // separador mínimo
                    double dx = aj.x - ai.x, dy = aj.y - ai.y, d = hypot(dx,dy);
                    if (d > 1e-12) {
                        double nxn = dx/d, nyn = dy/d, sep=1e-9;
                        ai.x -= nxn*sep; ai.y -= nyn*sep;
                        aj.x += nxn*sep; aj.y += nyn*sep;
                    }

                    stateStamp[ev.i]++; stateStamp[j]++;

                    writeEvent(out, String.format("PP i=%d j=%d", ai.id, aj.id));

                    // replan: i, j y watchers(i ∪ j)
                    Set<Integer> watchersI = new HashSet<>(dependents.get(ev.i)); 
                    Set<Integer> watchersJ = new HashSet<>(dependents.get(j));
                    recomputeNextAndEnqueue(ev.i); //recalculo las dos particulas
                    recomputeNextAndEnqueue(j);
                    if (this.verbose)
                        System.out.printf("[replan]  i=%d watchers=%d | j=%d watchers=%d\n",
                                ai.id, watchersI.size(), aj.id, watchersJ.size());
                    for (int k : watchersI) recomputeNextAndEnqueue(k); //recalculo los que dependen de i 
                    for (int k : watchersJ) if (k!=ev.i) recomputeNextAndEnqueue(k); // recalculo los que dependen de j (salteando i que ya lo hice)
                }

                writeSnapshot(out, "POST");
                if (T >= tMax) break;
            }

            writeSnapshot(out, "END");
            if (this.verbose) System.out.printf("== END   T=%.9f\n", T);
        }
    }

    // ---------- planificación ----------

    private void recomputeNextAndEnqueue(int i){
        // quitar dependencia vieja
        NextParticleEvent old = next[i]; 
        if (old.kind == Kind.PARTICLE && old.partner >= 0) {
            dependents.get(old.partner).remove(i); // aviso que i ya no depende del resto, porque voy a replanificar
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

        // dependencias
        if (nx.kind == Kind.PARTICLE && nx.partner >= 0) {
            dependents.get(nx.partner).add(i);
        }

        planStamp[i]++; // invalida el anterior
        if (Double.isFinite(nx.dt)) {
            pq.add(new Event(T + nx.dt, i, planStamp[i]));
        }

        if (this.verbose) {
            if (nx.kind == Kind.WALL && nx.wall != null) {
                System.out.printf("[plan]    T=%.9f  i=%d  next=WALL dt=%.9f  (bestWall=%.9f, bestPart=%.9f)  wall=(%.4f,%.4f)-(%.4f,%.4f)\n",
                        T, A.get(i).id, nx.dt, nx.bestWallDt, nx.bestPartDt,
                        nx.wall.x1, nx.wall.y1, nx.wall.x2, nx.wall.y2);
            } else if (nx.kind == Kind.PARTICLE) {
                System.out.printf("[plan]    T=%.9f  i=%d  next=PART dt=%.9f  (bestWall=%.9f, bestPart=%.9f)  j=%d\n",
                        T, A.get(i).id, nx.dt, nx.bestWallDt, nx.bestPartDt,
                        A.get(nx.partner).id);
            } else {
                System.out.printf("[plan]    T=%.9f  i=%d  next=NONE\n", T, A.get(i).id);
            }
        }
    }

    // ---------- logging a archivo ----------

    private void writeSnapshot(BufferedWriter out, String tag) throws IOException {
        out.write(String.format(java.util.Locale.US, "SNAPSHOT %s t=%.9f", tag, T));
        for (Agent a : A) {
            out.write(String.format(java.util.Locale.US,
                    " | id=%d x=%.6f y=%.6f vx=%.6f vy=%.6f r=%.6f",
                    a.id, a.x, a.y, a.vx, a.vy, a.r));
        }
        out.write("\n");
    }

    private void writeEvent(BufferedWriter out, String desc) throws IOException {
        out.write(String.format(java.util.Locale.US, "EVENT t=%.9f %s\n", T, desc));
    }

    // ---------- helpers ----------

    private static void microSeparate(Agent a, double nx, double ny){
        double sep = 1e-9; a.x += nx*sep; a.y += ny*sep;
    }
}
