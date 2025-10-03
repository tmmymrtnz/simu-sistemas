package simulation;

import java.util.Locale;

/**
 * Simple immutable 3D vector to keep the N-body code readable.
 */
public final class Vector3 {
    private final double x;
    private final double y;
    private final double z;

    public Vector3(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public static Vector3 zero() {
        return new Vector3(0.0, 0.0, 0.0);
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getZ() {
        return z;
    }

    public Vector3 add(Vector3 other) {
        return new Vector3(x + other.x, y + other.y, z + other.z);
    }

    public Vector3 subtract(Vector3 other) {
        return new Vector3(x - other.x, y - other.y, z - other.z);
    }

    public Vector3 scale(double factor) {
        return new Vector3(x * factor, y * factor, z * factor);
    }

    public double dot(Vector3 other) {
        return x * other.x + y * other.y + z * other.z;
    }

    public double normSquared() {
        return this.dot(this);
    }

    public double norm() {
        return Math.sqrt(normSquared());
    }

    public Vector3 normalize() {
        double length = norm();
        if (length == 0.0) {
            return Vector3.zero();
        }
        return scale(1.0 / length);
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "Vector3(%.6f, %.6f, %.6f)", x, y, z);
    }
}
