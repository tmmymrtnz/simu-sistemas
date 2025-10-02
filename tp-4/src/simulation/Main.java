package simulation;

import java.io.IOException;
import java.io.PrintWriter;

public class Main {

    public static void main(String[] args) {

        double tf =5.0;
        double dt = 0.000001;
        double t = 0.0;
        double x0 = 1.0;
        double k = 10000.0;
        double gamma = 100.0;
        double m = 70.0;
        double A = 1.0;
        double v0 = -A*gamma/(2.0*m);
        double VerletV0 = v0;
        double VerletX0 = x0;
        double GearX0 = x0;
        double GearV0 = v0;

        NumericMethod beeman = new Beeman(
            (x, v,time) -> (-k*x-gamma*v)/m,x0,v0
        );

        NumericMethod verlet = new OriginalVerlet(
            (x, v,time) -> (-k*x-gamma*v)/m,x0
        );


        NumericMethod gear5 = new GearFifthOrder(
                (x, v,time) -> (-k*x - gamma*v)/m, x0, v0, 0.0
        );


        try (PrintWriter pw = new PrintWriter("out.txt")) {
            pw.println("#t\tBeeman\tVerlet\tRealValue\tGear");
            while (t<tf){
                State BeemanState = beeman.solve(x0,v0,t,dt);
                State VerletState = verlet.solve(VerletX0,VerletV0,t,dt);
                State GearState = gear5.solve(GearX0,GearV0,t,dt);
                double realvalue = A * Math.exp(-gamma*t/(2.0*m)) * Math.cos(Math.sqrt(k/m - (gamma*gamma)/(4.0*m*m))*t);
                //print both values in a file
                pw.println(t + "\t" + BeemanState.getX() + "\t" + VerletState.getX() + "\t" + realvalue + "\t" + GearState.getX());
                x0 = BeemanState.getX();
                v0 = BeemanState.getV();
                VerletX0 = VerletState.getX();
                VerletV0 = VerletState.getV();
                GearX0 = GearState.getX();
                GearV0 = GearState.getV();
                t+=dt;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


}