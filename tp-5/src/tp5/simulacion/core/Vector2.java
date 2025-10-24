package tp5.simulacion.core;

/**
 * Lightweight immutable 2D vector record used across the pedestrian simulation.
 */
public record Vector2(double x, double y) {
    public static final Vector2 ZERO = new Vector2(0, 0);

    public double norm() {
        return Math.hypot(x, y);
    }

    public double normSq() {
        return x * x + y * y;
    }

    public Vector2 add(Vector2 other) {
        return new Vector2(x + other.x, y + other.y);
    }

    public Vector2 subtract(Vector2 other) {
        return new Vector2(x - other.x, y - other.y);
    }

    public Vector2 multiply(double scalar) {
        return new Vector2(x * scalar, y * scalar);
    }

    public Vector2 divide(double scalar) {
        return new Vector2(x / scalar, y / scalar);
    }

    public Vector2 normalized() {
        double n = norm();
        if (n == 0) {
            return ZERO;
        }
        return divide(n);
    }

    public double dot(Vector2 other) {
        return x * other.x + y * other.y;
    }
}
