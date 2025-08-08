package cim;

public class Particle {
    public final int id;
    public double x, y;
    public final double r;
    public double vx, vy; // para animación (opcional)

    public Particle(int id, double x, double y, double r) {
        this.id = id; this.x = x; this.y = y; this.r = r;
        this.vx = 0; this.vy = 0;
    }

    /** distancia centro-centro al cuadrado (mínima imagen si periódico) */
    public static double dist2(Particle a, Particle b, double L, boolean periodic) {
        double dx = Math.abs(a.x - b.x);
        double dy = Math.abs(a.y - b.y);
        if (periodic) {
            if (dx > 0.5 * L) dx = L - dx;
            if (dy > 0.5 * L) dy = L - dy;
        }
        return dx * dx + dy * dy;
    }
}
