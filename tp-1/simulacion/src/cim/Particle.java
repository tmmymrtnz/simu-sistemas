package cim;

public class Particle {
    public final int id;
    public final double x, y;
    public final double r;

    public Particle(int id, double x, double y, double r) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.r = r;
    }

    /** distancia centro-centro al cuadrado (con contorno periódico opcional) */
    public double dist2(Particle other, double L, boolean periodic) {
        double dx = Math.abs(x - other.x);
        double dy = Math.abs(y - other.y);
        if (periodic) {
            if (dx > 0.5 * L) dx = L - dx;
            if (dy > 0.5 * L) dy = L - dy;
        }
        return dx * dx + dy * dy;
    }
}
