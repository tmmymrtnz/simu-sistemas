package simulation;

import static java.lang.Math.*;

/** Mide presión promedio por recinto a partir de impulsos normales en paredes. */
public final class PressureTracker {
    private final double Lf;   // L_fixed
    private final double L;    // apertura (alto del pasillo)
    private final double eps;  // tolerancia geométrica

    // Perímetros efectivos (longitud de pared real) asumiento espesor unidad
    private final double perimLeft;   // cuadrado cerrado
    private final double perimRight;  // contorno real del pasillo

    private double impulseLeft = 0.0;
    private double impulseRight = 0.0;

    public PressureTracker(double L_fixed, double L) {
        this.Lf = L_fixed;
        this.L  = L;
        this.eps = 10 * Constants.TIME_EPS;

        // Recinto 1: cuadrado [0, Lf]x[0, Lf] cerrado (incluye pared imaginaria)
        this.perimLeft = 4.0 * Lf;

        // Recinto 2: contorno real del pasillo:
        //  horizontales (2*Lf) + pared derecha (L) + segmentos puertas (Lf-L) = 3*Lf
        this.perimRight = 3.0 * Lf;
    }

    // ---------- Clasificación geométrica ----------

    public boolean isInsideLeft(double x, double y) {
        return x >= -eps && x <= Lf + eps && y >= -eps && y <= Lf + eps;
    }

    public boolean isLeftWall(Wall w) {
        // bottom: y=0, x in [0,Lf]
        if (abs(w.y1 - 0.0) < eps && abs(w.y2 - 0.0) < eps &&
            min(w.x1, w.x2) <= Lf + eps && max(w.x1, w.x2) >= 0.0 - eps) return true;
        // top: y=Lf
        if (abs(w.y1 - Lf) < eps && abs(w.y2 - Lf) < eps &&
            min(w.x1, w.x2) <= Lf + eps && max(w.x1, w.x2) >= 0.0 - eps) return true;
        // left: x=0
        if (abs(w.x1 - 0.0) < eps && abs(w.x2 - 0.0) < eps &&
            min(w.y1, w.y2) <= Lf + eps && max(w.y1, w.y2) >= 0.0 - eps) return true;
        // La “pared imaginaria” NO existe como Wall: se registra vía cruce virtual.
        return false;
    }

    public boolean isRightWall(Wall w) {
        // puertas verticales en x=Lf, y in [0, (Lf-L)/2] y [ (Lf+L)/2, Lf ]
        double yLow = (Lf - L)/2.0;
        double yHigh= (Lf + L)/2.0;
        if (abs(w.x1 - Lf) < eps && abs(w.x2 - Lf) < eps) {
            double lo = min(w.y1, w.y2), hi = max(w.y1, w.y2);
            if (hi <= yLow + eps || lo >= yHigh - eps) return true; // segmentos de puerta
        }
        // horizontales del pasillo: y=yLow y y=yHigh entre x in [Lf, 2Lf]
        if (abs(w.y1 - yLow) < eps && abs(w.y2 - yLow) < eps &&
            min(w.x1, w.x2) >= Lf - eps && max(w.x1, w.x2) <= 2*Lf + eps) return true;
        if (abs(w.y1 - yHigh) < eps && abs(w.y2 - yHigh) < eps &&
            min(w.x1, w.x2) >= Lf - eps && max(w.x1, w.x2) <= 2*Lf + eps) return true;
        // pared derecha del pasillo: x=2Lf, y in [yLow, yHigh]
        if (abs(w.x1 - 2*Lf) < eps && abs(w.x2 - 2*Lf) < eps) {
            double lo = min(w.y1, w.y2), hi = max(w.y1, w.y2);
            if (lo >= yLow - eps && hi <= yHigh + eps) return true;
        }
        return false;
    }

    // ---------- Registro de impulsos ----------

    /** Impulso por colisión real (pared). Se suma a la región correspondiente. */
    public void addRealWallImpulse(Wall w, double impulseAbs) {
        if (impulseAbs <= 0) return;
        if (isLeftWall(w)) impulseLeft += impulseAbs;
        else if (isRightWall(w)) impulseRight += impulseAbs;
    }

    /** Impulso “virtual” en la pared imaginaria del recinto izquierdo (x=Lf). */
    public void addVirtualLeftImpulse(double impulseAbs) {
        if (impulseAbs > 0) impulseLeft += impulseAbs;
    }

    // ---------- Cálculo de presión ----------

    public double pressureLeft(double time) {
        double t = max(time, 1e-12);
        return impulseLeft / (perimLeft * t);
    }

    public double pressureRight(double time) {
        double t = max(time, 1e-12);
        return impulseRight / (perimRight * t);
    }

    public double getPerimLeft()  { return perimLeft; }
    public double getPerimRight() { return perimRight; }

    public double getImpulseLeft(){ return impulseLeft; }
    public double getImpulseRight(){ return impulseRight; }
}
