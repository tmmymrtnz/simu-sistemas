package simulation;

import static java.lang.Math.*;

public final class Collisions {

    private Collisions(){}

    /** t>=0 hasta colisión partícula–partícula; INF si no hay. */
    public static double timeToParticle(Agent a, Agent b) {
        final double EPS = Constants.TIME_EPS;

        double rx = b.x - a.x, ry = b.y - a.y;
        double vx = b.vx - a.vx, vy = b.vy - a.vy;
        double R  = a.r + b.r;

        double A = vx*vx + vy*vy;
        double B = 2.0*(rx*vx + ry*vy);
        double C = rx*rx + ry*ry - R*R;

        if (A < EPS) return Double.POSITIVE_INFINITY;
        double D = B*B - 4*A*C;
        if (D < 0) return Double.POSITIVE_INFINITY;

        // acercándose al inicio: r·v < 0
        if ((rx*vx + ry*vy) >= 0) return Double.POSITIVE_INFINITY;

        double sqrtD = sqrt(D);
        double t1 = (-B - sqrtD)/(2*A);
        double t2 = (-B + sqrtD)/(2*A);

        if (t1 >= EPS) return t1;
        if (t2 >= EPS) return t2;
        return Double.POSITIVE_INFINITY;
    }

    /** t>=0 hasta colisión partícula–pared (segmento); INF si no hay. */
    public static double timeToWall(Agent a, Wall w) {
        final double EPS = Constants.TIME_EPS;

        double ux = w.x2 - w.x1, uy = w.y2 - w.y1;
        double L2 = ux*ux + uy*uy;
        if (L2 < EPS) return timeToPoint(a, w.x1, w.y1);

        // |cross((c0 + v t), u)| = r|u|
        double c0x = a.x - w.x1, c0y = a.y - w.y1;
        double cu0 = c0x*uy - c0y*ux;
        double cv  = a.vx*uy - a.vy*ux;
        double rhs = a.r * sqrt(L2);

        double best = Double.POSITIVE_INFINITY;

        if (abs(cv) > EPS) {
            double tA = ( rhs - cu0)/cv;
            double tB = (-rhs - cu0)/cv;
            double candA = checkWallSegmentHit(a, w, tA);
            if (candA < best) best = candA;
            double candB = checkWallSegmentHit(a, w, tB);
            if (candB < best) best = candB;
        }

        // extremos
        double tp1 = timeToPoint(a, w.x1, w.y1);
        if (tp1 < best) best = tp1;
        double tp2 = timeToPoint(a, w.x2, w.y2);
        if (tp2 < best) best = tp2;

        return best;
    }

    /** Normal unitaria pared→partícula en el instante actual (post-avance al evento). */
    public static double[] wallImpactNormal(Agent a, Wall w) {
        final double EPS = Constants.TIME_EPS;

        double ux = w.x2 - w.x1, uy = w.y2 - w.y1;
        double L2 = ux*ux + uy*uy;
        if (L2 < EPS) {
            double dx = a.x - w.x1, dy = a.y - w.y1;
            double d = hypot(dx, dy);
            if (d < EPS) return new double[]{1,0};
            return new double[]{dx/d, dy/d};
        }
        double wx = a.x - w.x1, wy = a.y - w.y1;
        double s = (wx*ux + wy*uy)/L2;
        s = max(0.0, min(1.0, s));
        double px = w.x1 + s*ux, py = w.y1 + s*uy;
        double nx = a.x - px, ny = a.y - py;
        double d  = hypot(nx, ny);
        if (d < EPS) {
            double nnx = -uy, nny = ux;
            double inv = 1.0 / max(EPS, hypot(nnx, nny));
            nnx *= inv; nny *= inv;
            if (a.vx*nnx + a.vy*nny >= 0) { nnx = -nnx; nny = -nny; }
            return new double[]{nnx, nny};
        }
        return new double[]{nx/d, ny/d};
    }

    /** Reflejo elástico contra pared. */
    public static void resolveParticleWall(Agent a, double nx, double ny) {
        double vn = a.vx*nx + a.vy*ny;
        if (vn < 0) {
            a.vx -= 2*vn*nx;
            a.vy -= 2*vn*ny;
        }
    }

    /** Colisión elástica (e=1), masas iguales, robusta con el mismo signo que la detección. */
    public static void resolveParticleParticle(Agent a, Agent b) {
        final double EPS = Constants.TIME_EPS;

        double dx = b.x - a.x, dy = b.y - a.y;
        double d  = hypot(dx, dy);
        if (d < EPS) return;

        double nx = dx/d, ny = dy/d;

        // Usar el mismo relativo que la detección: (v_a - v_b)·n
        double rel = (a.vx - b.vx)*nx + (a.vy - b.vy)*ny;

        // Si rel <= 0 están separándose o tangenciales: no hay impulso
        if (rel <= 0) return;

        // Impulso para m1=m2, e=1: v_a' = v_a - rel n ; v_b' = v_b + rel n
        a.vx -= rel * nx; a.vy -= rel * ny;
        b.vx += rel * nx; b.vy += rel * ny;
    }

    // ---------- privados ----------

    private static double checkWallSegmentHit(Agent a, Wall w, double t) {
        final double EPS = Constants.TIME_EPS;

        if (!(t >= EPS && Double.isFinite(t))) return Double.POSITIVE_INFINITY;
        double cx = a.x + a.vx*t, cy = a.y + a.vy*t;
        double ux = w.x2 - w.x1,  uy = w.y2 - w.y1;
        double L2 = ux*ux + uy*uy;
        double wx = cx - w.x1, wy = cy - w.y1;
        double s  = (wx*ux + wy*uy)/L2;
        if (s < -1e-12 || s > 1.0 + 1e-12) return Double.POSITIVE_INFINITY;

        // evitar tangencial pura
        double nx = -uy, ny = ux;
        double vn = a.vx*nx + a.vy*ny;
        if (abs(vn) < 1e-14) return Double.POSITIVE_INFINITY;

        return t;
    }

    private static double timeToPoint(Agent a, double qx, double qy) {
        final double EPS = Constants.TIME_EPS;

        double rx = a.x - qx, ry = a.y - qy;
        double A = a.vx*a.vx + a.vy*a.vy;
        if (A < EPS) return Double.POSITIVE_INFINITY;
        double B = 2.0*(rx*a.vx + ry*a.vy);
        double C = rx*rx + ry*ry - a.r*a.r;

        double D = B*B - 4*A*C;
        if (D < 0) return Double.POSITIVE_INFINITY;
        if (B >= 0) return Double.POSITIVE_INFINITY; // alejándose

        double t1 = (-B - sqrt(D))/(2*A);
        double t2 = (-B + sqrt(D))/(2*A);
        if (t1 >= EPS) return t1;
        if (t2 >= EPS) return t2;
        return Double.POSITIVE_INFINITY;
    }
}
