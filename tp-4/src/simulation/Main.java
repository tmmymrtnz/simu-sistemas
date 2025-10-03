package simulation;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.io.File;

public class Main {

    public static void main(String[] args) {

        File outDir = new File("src/out/simulation");
        if (!outDir.exists()) {
            outDir.mkdirs();
        }

        double tf = 5.0;
        double x0 = 1.0;
        double k = 10000.0;
        double gamma = 100.0;
        double m = 70.0;
        double A = 1.0;
        double v0 = -A * gamma / (2.0 * m);

        List<Double> dtValues = Arrays.asList(1e-2, 1e-3, 1e-4, 1e-5, 1e-6);
        for (double dt : dtValues) {
            runSimulation(dt, tf, x0, v0, k, gamma, m, A, String.format("src/out/simulation/out_%.0e.txt", dt));
        }
        double fullPlotDt = 0.01;
        runSimulation(fullPlotDt, tf, x0, v0, k, gamma, m, A, "src/out/simulation/out_full.txt");
    }

    private static void runSimulation(double dt, double tf, double x0, double v0, double k, double gamma, double m, double A, String outputFileName) {
        double t = 0.0;
        double BeemanX0 = x0;
        double BeemanV0 = v0;
        double VerletX0 = x0;
        double VerletV0 = v0;
        double GearX0 = x0;
        double GearV0 = v0;

        NumericMethod beeman = new Beeman(
            (x, v, time) -> (-k * x - gamma * v) / m, BeemanX0, BeemanV0
        );

        NumericMethod verlet = new OriginalVerlet(
            (x, v, time) -> (-k * x - gamma * v) / m, VerletX0
        );

        NumericMethod gear5 = new GearFifthOrder(
            (x, v, time) -> (-k * x - gamma * v) / m, GearX0, GearV0, 0.0
        );

        try (PrintWriter pw = new PrintWriter(outputFileName)) {
            pw.println("#t\tBeeman\tVerlet\tRealValue\tGear");
            while (t < tf) {
                State BeemanState = beeman.solve(BeemanX0, BeemanV0, t, dt);
                State VerletState = verlet.solve(VerletX0, VerletV0, t, dt);
                State GearState = gear5.solve(GearX0, GearV0, t, dt);
                double realvalue = A * Math.exp(-gamma * t / (2.0 * m)) * Math.cos(Math.sqrt(k / m - (gamma * gamma) / (4.0 * m * m)) * t);
                
                pw.println(t + "\t" + BeemanState.getX() + "\t" + VerletState.getX() + "\t" + realvalue + "\t" + GearState.getX());
                
                BeemanX0 = BeemanState.getX();
                BeemanV0 = BeemanState.getV();
                VerletX0 = VerletState.getX();
                VerletV0 = VerletState.getV();
                GearX0 = GearState.getX();
                GearV0 = GearState.getV();
                t += dt;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Generated file: " + outputFileName);
    }
}