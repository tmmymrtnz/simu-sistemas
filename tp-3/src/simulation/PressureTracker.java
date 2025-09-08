package simulation;

import static java.lang.Math.*;
import java.util.Arrays;

/** Presión por recinto = (Σ impulsos reales en paredes) / (dt · Σ longitudes) */
public final class PressureTracker {
    // Geometría
    private final double Lf;   // L_fixed
    private final double L;    // apertura (alto del pasillo)
    private final double yLow, yHigh;
    private final double eps;

    // Identificadores de paredes físicas
    private enum PW {
        L_BOTTOM, L_LEFT, L_TOP, L_DOOR_LOW, L_DOOR_HIGH,   // recinto izquierdo
        R_LOW, R_HIGH, R_RIGHT                               // recinto derecho
    }

    // Longitud de cada pared
    private final double[] wLen = new double[PW.values().length];

    // Impulsos por-intervalo (por pared)
    private final double[] impCurr = new double[PW.values().length];

    // Impulsos acumulados (por pared, desde t=0)
    private final double[] impCum  = new double[PW.values().length];

    // Conjuntos de paredes por recinto (índices)
    private final int[] leftWalls  = new int[]{ idx(PW.L_BOTTOM), idx(PW.L_LEFT), idx(PW.L_TOP),
                                               idx(PW.L_DOOR_LOW), idx(PW.L_DOOR_HIGH) };
    private final int[] rightWalls = new int[]{ idx(PW.R_LOW), idx(PW.R_HIGH), idx(PW.R_RIGHT) };

    // Estado de muestreo
    private double lastT = 0.0;
    private double lastPleft = 0.0, lastPright = 0.0;

    private static int idx(PW w){ return w.ordinal(); }

    public PressureTracker(double L_fixed, double L) {
        this.Lf = L_fixed;
        this.L  = L;
        this.yLow  = (Lf - L) / 2.0;
        this.yHigh = (Lf + L) / 2.0;
        this.eps = 10.0 * Constants.TIME_EPS;

        // Longitudes geométricas
        wLen[idx(PW.L_BOTTOM)]   = Lf;
        wLen[idx(PW.L_LEFT)]     = Lf;
        wLen[idx(PW.L_TOP)]      = Lf;

        double doorLen = max(0.0, (Lf - L) / 2.0);
        wLen[idx(PW.L_DOOR_LOW)]  = doorLen;
        wLen[idx(PW.L_DOOR_HIGH)] = doorLen;

        wLen[idx(PW.R_LOW)]    = Lf;
        wLen[idx(PW.R_HIGH)]   = Lf;
        wLen[idx(PW.R_RIGHT)]  = max(0.0, L);
    }

    // --------- getters geométricos ----------
    public double getLf()   { return Lf; }
    public double getYLow(){ return yLow; }
    public double getYHigh(){ return yHigh; }

    // Perímetros efectivos por recinto (solo paredes con longitud > 0)
    public double getPerimLeft()  { return totalLength(leftWalls); }
    public double getPerimRight() { return totalLength(rightWalls); }

    // --------- pertenencia (utilidad externa) ----------
    public boolean isInsideLeft(double x, double y){
        return x >= -eps && x <= Lf + eps && y >= -eps && y <= Lf + eps;
    }
    public boolean isInsideRight(double x, double y){
        return x >= Lf - eps && x <= 2*Lf + eps && y >= yLow - eps && y <= yHigh + eps;
    }

    // --------- clasificación de paredes físicas ----------
    private Integer classifyWall(Wall w){
        final double MID_EPS = 1e-12; // margen para decidir por punto medio
        boolean horiz = abs(w.y1 - w.y2) < eps;
        boolean vert  = abs(w.x1 - w.x2) < eps;

        double xm = 0.5 * (w.x1 + w.x2);
        double ym = 0.5 * (w.y1 + w.y2);

        if (horiz) {
            // y=0 (puede ser piso izq. o borde inferior del pasillo si L=Lf)
            if (abs(w.y1 - 0.0) < eps) {
                if (xm < Lf - MID_EPS) return idx(PW.L_BOTTOM);
                if (xm > Lf + MID_EPS) return idx(PW.R_LOW); // solo aplica cuando yLow=0 (L=Lf)
                // Si cae justo en xm≈Lf, decide por rango: [0,Lf] -> izquierda; [Lf,2Lf] -> derecha
                double xlo = min(w.x1, w.x2), xhi = max(w.x1, w.x2);
                return (xhi <= Lf + eps) ? idx(PW.L_BOTTOM) : idx(PW.R_LOW);
            }
            // y=Lf (techo izq. o borde superior del pasillo si L=Lf)
            if (abs(w.y1 - Lf) < eps) {
                if (xm < Lf - MID_EPS) return idx(PW.L_TOP);
                if (xm > Lf + MID_EPS) return idx(PW.R_HIGH); // solo aplica cuando yHigh=Lf (L=Lf)
                double xlo = min(w.x1, w.x2), xhi = max(w.x1, w.x2);
                return (xhi <= Lf + eps) ? idx(PW.L_TOP) : idx(PW.R_HIGH);
            }
            // Pasillo horizontal general: y=yLow o y=yHigh con x en (Lf, 2Lf)
            if (abs(w.y1 - yLow)  < eps && xm > Lf + MID_EPS) return idx(PW.R_LOW);
            if (abs(w.y1 - yHigh) < eps && xm > Lf + MID_EPS) return idx(PW.R_HIGH);
        }

        if (vert) {
            // x=0 (pared izquierda del cuadrado)
            if (abs(w.x1 - 0.0) < eps && ym >= -eps && ym <= Lf + eps) return idx(PW.L_LEFT);
            // x=Lf (dos "puertas" del cuadrado izq.)
            if (abs(w.x1 - Lf) < eps) {
                double lo = min(w.y1, w.y2), hi = max(w.y1, w.y2);
                if (hi <= yLow + eps) return idx(PW.L_DOOR_LOW);
                if (lo >= yHigh - eps) return idx(PW.L_DOOR_HIGH);
            }
            // x=2Lf (pared derecha del pasillo)
            if (abs(w.x1 - 2*Lf) < eps) {
                double lo = min(w.y1, w.y2), hi = max(w.y1, w.y2);
                if (lo >= yLow - eps && hi <= yHigh + eps) return idx(PW.R_RIGHT);
            }
        }

        return null; // no nos interesa (o numéricamente ambigua)
    }

    // --------- registro de impulsos (solo REALES) ----------
    public void addRealWallImpulse(Wall w, double impulseAbs) {
        if (impulseAbs <= 0) return;
        Integer k = classifyWall(w);
        if (k != null) {
            impCurr[k] += impulseAbs;
            impCum[k]  += impulseAbs;
        }
    }

    // --------- helpers de longitud/sumas ----------
    private double totalLength(int[] walls){
        double Lsum = 0.0;
        for (int k : walls) if (wLen[k] > eps) Lsum += wLen[k];
        return Lsum;
    }
    private double sumImpulses(double[] J, int[] walls){
        double s = 0.0;
        for (int k : walls) s += J[k];
        return s;
    }

    // --------- presiones por-intervalo (dt = T - lastT) ----------
    public double[] sampleInterval(double T){
        double dt = max(Constants.TIME_EPS, T - lastT);

        double pL = lengthWeightedPressureInterval(leftWalls, dt);
        double pR = lengthWeightedPressureInterval(rightWalls, dt);

        Arrays.fill(impCurr, 0.0);
        lastT = T;
        lastPleft  = pL;
        lastPright = pR;
        return new double[]{ pL, pR };
    }

    private double lengthWeightedPressureInterval(int[] walls, double dt){
        double Lsum = totalLength(walls);
        if (Lsum <= eps) return 0.0;
        double Jsum = sumImpulses(impCurr, walls);
        return Jsum / (Lsum * dt);
    }

    public double pressureLeftIntervalCached()  { return lastPleft; }
    public double pressureRightIntervalCached() { return lastPright; }

    // --------- presiones acumuladas (desde t=0) ----------
    public double pressureLeftCumulative(double T){
        double t = max(T, Constants.TIME_EPS);
        return lengthWeightedPressureCumulative(leftWalls, t);
    }
    public double pressureRightCumulative(double T){
        double t = max(T, Constants.TIME_EPS);
        return lengthWeightedPressureCumulative(rightWalls, t);
    }

    private double lengthWeightedPressureCumulative(int[] walls, double t){
        double Lsum = totalLength(walls);
        if (Lsum <= eps) return 0.0;
        double Jsum = sumImpulses(impCum, walls);
        return Jsum / (Lsum * t);
    }
}
