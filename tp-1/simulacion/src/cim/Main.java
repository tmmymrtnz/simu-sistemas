package cim;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 *  java -jar cim.jar [archivo.properties]
 *  (si se omite, busca "config.properties" en cwd)
 */
public class Main {

    private static Properties load(String path) throws IOException {
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
        String file = (args.length == 0) ? "config.properties" : args[0];
        Properties P = load(file);

        int     N  = req(P,"N", Integer::parseInt);
        double  L  = req(P,"L", Double::parseDouble);
        int     M  = req(P,"M", Integer::parseInt);
        double  rc = req(P,"rc",Double::parseDouble);
        double  r  = req(P,"r", Double::parseDouble);
        boolean periodic = Boolean.parseBoolean(P.getProperty("periodic","false"));
        long    seed     = P.containsKey("seed") ? Long.parseLong(P.getProperty("seed")) : -1;
        String  outBase  = P.getProperty("outputBase","run");

        Random rnd = (seed >= 0) ? new Random(seed) : new Random();

        /* generar partículas (rechazo simple para no superponer) */
        List<Particle> ps = new ArrayList<>(N);
        for (int id = 0; id < N; id++) {
            double x, y;  int tries = 0;  boolean ok;
            do {
                x = rnd.nextDouble() * L;  y = rnd.nextDouble() * L;  ok = true;
                for (Particle p : ps) {
                    double dx = x - p.x, dy = y - p.y;
                    if (periodic) {
                        if (dx > 0.5*L) dx -= L;  if (dx < -0.5*L) dx += L;
                        if (dy > 0.5*L) dy -= L;  if (dy < -0.5*L) dy += L;
                    }
                    if (dx*dx + dy*dy < (2*r)*(2*r)) { ok = false; break; }
                }
            } while (!ok && ++tries < 1_000);
            ps.add(new Particle(id,x,y,r));
        }

        /* CIM */
        CellIndexMethod cim = new CellIndexMethod(L,rc,M,periodic);
        long t0 = System.nanoTime();
        var neigh = cim.neighbours(ps);
        long dt = System.nanoTime() - t0;

        /* salida */
        Path dir = Path.of("out", outBase);  Files.createDirectories(dir);
        try (BufferedWriter w = Files.newBufferedWriter(dir.resolve("neighbours.txt"))) {
            for (Particle p : ps) { w.write(p.id + ": " + neigh.get(p.id)); w.newLine(); }
        }
        try (BufferedWriter w = Files.newBufferedWriter(dir.resolve("particles.csv"))) {
            w.write("id,x,y,r\n");
            for (Particle p : ps)
                w.write(String.format(Locale.US,"%d,%.6f,%.6f,%.3f%n",p.id,p.x,p.y,p.r));
        }

        System.out.printf(Locale.US,"CIM %.3f ms — resultados en %s%n", dt/1e6, dir);
    }
}
