package flock;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import static flock.Props.req;

public class Main {
    public static void main(String[] args) throws IOException {
        String propFile = (args.length==0) ? "simulacion/flock.properties" : args[0];
        var P = Props.load(propFile);

        int     N   = req(P,"N", Integer::parseInt);
        double  L   = req(P,"L", Double::parseDouble);
        double  R   = req(P,"R", Double::parseDouble);
        double  v0  = req(P,"v0", Double::parseDouble);
        double  dt  = req(P,"dt", Double::parseDouble);
        int     steps = req(P,"steps", Integer::parseInt);
        double  eta = Double.parseDouble(P.getProperty("eta","0.0"));
        boolean periodic = Boolean.parseBoolean(P.getProperty("periodic","true"));
        Integer Mopt = P.containsKey("M") ? Integer.parseInt(P.getProperty("M")) : null;
        boolean includeSelf = Boolean.parseBoolean(P.getProperty("includeSelf","true"));
        String  rule = P.getProperty("rule","vicsek").toLowerCase(); // "vicsek" | "voter"
        long    seed = P.containsKey("seed") ? Long.parseLong(P.getProperty("seed")) : 12345L;
        String  outBase = P.getProperty("outputBase", "run_vicsek");

        Random rnd = new Random(seed);

        // Inicialización
        ArrayList<Agent> agents = new ArrayList<>(N);
        for (int i=0;i<N;i++){
            double x = rnd.nextDouble()*L;
            double y = rnd.nextDouble()*L;
            double th= rnd.nextDouble()*2*Math.PI;
            agents.add(new Agent(i,x,y,th,v0));
        }

        
        Grid grid = new Grid(L, R, Mopt, periodic);
        UpdateRule updater = rule.equals("voter") ? new VoterUpdate() : new VicsekUpdate();


        Path dir;
        if (outBase != null && !outBase.isEmpty()) {
            dir = Path.of("out", outBase);
        } else {
            String params = String.format("N%d_L%.1f_R%.2f_eta%.2f_rule_%s", N, L, R, eta, rule);
            String idrandom = UUID.randomUUID().toString().substring(0, 8);
            dir = Path.of("out", "simulacion_" + params + "_" + idrandom);
        }
        System.out.println("Directorio de salida: " + dir.toAbsolutePath());
        Files.createDirectories(dir);

        IOUtil.writeStatic(dir, N,L,v0,R,eta,dt,steps,periodic,rule,seed,Mopt);

        Path obsFile = dir.resolve("observables.csv");

        // t=0
        grid.rebuild(agents);
        IOUtil.writeFrame(dir, 0, agents);
        IOUtil.appendObs(obsFile, 0, IOUtil.polarization(agents));

        for (int t=1; t<=steps; t++){
            // actualizar ángulos
            updater.updateThetas(agents, grid, eta, rnd, includeSelf);

            // avanzar posiciones + contorno
            for (Agent a: agents){
                a.x += a.vx()*dt; a.y += a.vy()*dt;
                if (periodic){
                    if (a.x < 0) a.x += L; if (a.x >= L) a.x -= L;
                    if (a.y < 0) a.y += L; if (a.y >= L) a.y -= L;
                } else {
                    if (a.x < 0) a.x = 0; if (a.x > L) a.x = L;
                    if (a.y < 0) a.y = 0; if (a.y > L) a.y = L;
                }
            }

            grid.rebuild(agents);
            IOUtil.writeFrame(dir, t, agents);
            IOUtil.appendObs(obsFile, t, IOUtil.polarization(agents));
        }

        Files.copy(dir.resolve(String.format("t%04d.txt", steps)),
                   dir.resolve("last.txt"),
                   java.nio.file.StandardCopyOption.REPLACE_EXISTING);

        System.out.printf(java.util.Locale.US,
          "✅ Listo. rule=%s N=%d L=%.1f R=%.2f eta=%.2f steps=%d periodic=%s | out=%s | M=%d%n",
          rule,N,L,R,eta,steps,periodic,dir, grid.M);
    }
}
