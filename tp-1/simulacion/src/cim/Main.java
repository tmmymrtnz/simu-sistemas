package cim;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 *  java -jar simulacion/cim.jar [config.properties]
 *  Salidas por frame: out/<base>/particles_t####.csv, neighbours_t####.txt
 *  Copias: particles.csv / neighbours.txt (último frame)
 *  metrics.csv: tiempos y comparaciones (solo CIM).
 */
public class Main {

    private static Properties loadProps(String path) throws IOException {
        Properties p = new Properties();
        try (var in = Files.newInputStream(Path.of(path))) { p.load(in); }
        return p;
    }
    private static <T> T req(Properties p, String k, java.util.function.Function<String,T> f) {
        String v = p.getProperty(k);
        if (v == null) throw new IllegalArgumentException("Falta propiedad: " + k);
        return f.apply(v.trim());
    }

    public static void main(String[] args) throws IOException {
        String propFile = (args.length == 0) ? "config.properties" : args[0];
        Properties P = loadProps(propFile);

        int     N   = req(P,"N", Integer::parseInt);
        double  L   = req(P,"L", Double::parseDouble);
        int     M   = req(P,"M", Integer::parseInt);
        double  rc  = req(P,"rc",Double::parseDouble);
        boolean periodic = Boolean.parseBoolean(P.getProperty("periodic","true")); // tu default
        long    seed     = P.containsKey("seed") ? Long.parseLong(P.getProperty("seed")) : -1;
        String  outBase  = P.getProperty("outputBase","run");

        // NUEVO: modo de interacción
        boolean useRadii = Boolean.parseBoolean(P.getProperty("useRadii","false"));
        // Si usás radios, podés definir r fijo o rMin/rMax; si no, los forzamos a 0.

        boolean hasRange = P.containsKey("rMin") && P.containsKey("rMax");
        double rFixed = P.containsKey("r") ? Double.parseDouble(P.getProperty("r")) : 0.0;
        double rMin = hasRange ? Double.parseDouble(P.getProperty("rMin")) : rFixed;
        double rMax = hasRange ? Double.parseDouble(P.getProperty("rMax")) : rFixed;

        Random rnd = (seed >= 0) ? new Random(seed) : new Random();

        // generar partículas (rechazo simple solo si useRadii=true)
        List<Particle> ps = new ArrayList<>(N);
        for (int id = 0; id < N; id++) {
            double r = useRadii ? (hasRange ? (rMin + rnd.nextDouble()*(rMax - rMin)) : rFixed) : 0.0;
            double x, y; int tries=0; boolean ok;
            do {
                x = rnd.nextDouble()*L; y = rnd.nextDouble()*L; ok=true;
                if (useRadii) {
                    for (Particle p : ps) {
                        double dx = x - p.x, dy = y - p.y;
                        if (periodic) {
                            if (dx > 0.5*L) dx -= L; if (dx < -0.5*L) dx += L;
                            if (dy > 0.5*L) dy -= L; if (dy < -0.5*L) dy += L;
                        }
                        double minD = (r + p.r);
                        if (dx*dx + dy*dy < (minD*minD)) { ok=false; break; }
                    }
                }
            } while (!ok && ++tries < 2000);
            Particle np = new Particle(id,x,y,r);
            // Velocidades opcionales para animación
            double vmax = Double.parseDouble(P.getProperty("vmax","0.0"));
            if (vmax > 0) { np.vx = (rnd.nextDouble()*2-1)*vmax; np.vy = (rnd.nextDouble()*2-1)*vmax; }
            ps.add(np);
        }

        // salida
        Path dir = Path.of("out", outBase);
        Files.createDirectories(dir);

        // métrica por frame
        List<String> metrics = new ArrayList<>();
        metrics.add("frame,method,ms,comparisons,neighbors_total");

        int frames = Integer.parseInt(P.getProperty("frames","1"));
        double dt  = Double.parseDouble(P.getProperty("dt","0.1"));

        for (int f=0; f<frames; f++) {
            CellIndexMethod cim = new CellIndexMethod(L, rc, M, periodic, useRadii);
            long t0 = System.nanoTime();
            Map<Integer, Set<Integer>> neigh = cim.neighbours(ps);
            long dtCim = System.nanoTime() - t0;

            long neighborsTotal = neigh.values().stream().mapToLong(Set::size).sum();

            metrics.add(String.format(java.util.Locale.US,
                "%d,CIM,%.3f,%d,%d", f, dtCim/1e6, cim.comparisons, neighborsTotal));

            String tag = String.format("t%04d", f);
            writeParticles(dir.resolve("particles_"+tag+".csv"), ps);
            writeNeighbours(dir.resolve("neighbours_"+tag+".txt"), neigh);

            if (f < frames-1) step(ps, L, dt, periodic);
        }

        String last = String.format("t%04d", frames-1);
        Files.copy(dir.resolve("particles_"+last+".csv"), dir.resolve("particles.csv"),
                   java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        Files.copy(dir.resolve("neighbours_"+last+".txt"), dir.resolve("neighbours.txt"),
                   java.nio.file.StandardCopyOption.REPLACE_EXISTING);

        Files.write(dir.resolve("metrics.csv"), metrics);

        System.out.printf(java.util.Locale.US,
            "✅ Listo. Frames=%d | useRadii=%s | periodic=%s | Resultados en %s%n",
            frames, useRadii, periodic, dir);
    }

    private static void step(List<Particle> ps, double L, double dt, boolean periodic) {
        for (Particle p : ps) {
            p.x += p.vx * dt; p.y += p.vy * dt;
            if (periodic) {
                if (p.x < 0) p.x += L; if (p.x >= L) p.x -= L;
                if (p.y < 0) p.y += L; if (p.y >= L) p.y -= L;
            } else {
                if (p.x < p.r) { p.x = p.r; p.vx = -p.vx; }
                if (p.x > L - p.r) { p.x = L - p.r; p.vx = -p.vx; }
                if (p.y < p.r) { p.y = p.r; p.vy = -p.vy; }
                if (p.y > L - p.r) { p.y = L - p.r; p.vy = -p.vy; }
            }
        }
    }

    private static void writeParticles(Path file, List<Particle> ps) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(file)) {
            w.write("id,x,y,r,vx,vy\n");
            for (Particle p : ps) {
                w.write(String.format(java.util.Locale.US,
                    "%d,%.6f,%.6f,%.6f,%.6f,%.6f%n", p.id,p.x,p.y,p.r,p.vx,p.vy));
            }
        }
    }
    private static void writeNeighbours(Path file, Map<Integer, Set<Integer>> neigh) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(file)) {
            for (var e : neigh.entrySet()) {
                String list = e.getValue().stream().sorted().map(Object::toString)
                               .collect(Collectors.joining(","));
                w.write(e.getKey() + ": [" + list + "]");
                w.newLine();
            }
        }
    }
}
