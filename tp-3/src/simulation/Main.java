package simulation;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

public class Main {

    private static void usage() {
        System.out.println("Uso: java -jar sim.jar <N> <L>");
        System.out.println("  N: cantidad de partículas (int)");
        System.out.println("  L: apertura del pasillo (double, metros)");
        System.out.println("Ej: java -jar sim.jar 12 0.05");
    }

    public static void main(String[] args) throws Exception {
        int N = 12;
        double L = 0.05;

        if (args.length >= 2) {
            try {
                N = Integer.parseInt(args[0]);
                L = Double.parseDouble(args[1]);
            } catch (Exception e) {
                usage();
                System.exit(1);
            }
        } else {
            System.out.println("[warn] Parámetros no provistos. Usando por defecto: N=" + N + " L=" + L);
            System.out.println("       (Sugerido) " +
                    "java -jar sim.jar <N> <L>");
        }

        Map map = new Map(L);
        map.createParticles(N);

        List<Agent> agents = map.particles;
        List<Wall>  walls  = map.walls;

        // Nombres de salida con L y N en el nombre
        String Lstr = String.format(Locale.US, "%.3f", map.L);
        String Nstr = Integer.toString(N);

        Path eventsPath   = Path.of("events_L=" + Lstr + "_N=" + Nstr + ".txt");
        Path pressurePath = Path.of("pressure_L=" + Lstr + "_N=" + Nstr + ".txt");

        TargetEventSim sim = new TargetEventSim(agents, walls, map.L_fixed, map.L);

        // tMax es una cota dura; además el simulador corta cuando P_left~P_right durante STEADY_SAMPLES
        double tMax = 60.0;
        // verbose=true para prints en consola
        sim.run(tMax, eventsPath, pressurePath, /*verbose=*/true);

        System.out.println("Listo. Logs:");
        System.out.println(" - " + eventsPath.toAbsolutePath());
        System.out.println(" - " + pressurePath.toAbsolutePath());
    }
}
