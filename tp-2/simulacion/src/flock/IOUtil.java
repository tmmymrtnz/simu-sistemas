package flock;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class IOUtil {
    public static void writeStatic(Path dir, int N, double L, double v0, double R, double eta,
                                   double dt, int steps, boolean periodic, String rule, long seed, Integer M){
        List<String> lines = new ArrayList<>();
        lines.add("N " + N);
        lines.add("L " + L);
        lines.add("v0 " + v0);
        lines.add("R " + R);
        lines.add("eta " + eta);
        lines.add("dt " + dt);
        lines.add("steps " + steps);
        lines.add("periodic " + periodic);
        lines.add("rule " + rule);
        lines.add("seed " + seed);
        lines.add("M " + (M==null? -1 : M));
        try { Files.write(dir.resolve("static.txt"), lines); }
        catch (IOException e){ throw new RuntimeException(e); }
    }

    /** Frame: columnas: id x y vx vy theta */
    public static void writeFrame(Path dir, int t, List<Agent> as) {
        Path f = dir.resolve(String.format("t%04d.txt", t));
        try (BufferedWriter w = Files.newBufferedWriter(f)) {
            w.write("id x y vx vy theta\n");
            for (Agent a: as) {
                w.write(String.format(java.util.Locale.US,
                    "%d %.6f %.6f %.6f %.6f %.6f%n", a.id, a.x, a.y, a.vx(), a.vy(), a.theta));
            }
        } catch (IOException e) { throw new RuntimeException(e); }
    }

    /** va = (1/N) || sum (cos theta, sin theta) || */
    public static double polarization(List<Agent> as){
        double Sx=0, Sy=0;
        for (Agent a: as){ Sx += Math.cos(a.theta); Sy += Math.sin(a.theta); }
        double mod = Math.sqrt(Sx*Sx + Sy*Sy);
        return (mod / as.size());
    }

    public static void appendObs(Path f, int t, double va) {
        try {
            if (!Files.exists(f)) Files.writeString(f, "t,va\n");
            Files.writeString(f, String.format(java.util.Locale.US, "%d,%.6f%n", t, va),
                StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        } catch (IOException e) { throw new RuntimeException(e); }
    }
}
