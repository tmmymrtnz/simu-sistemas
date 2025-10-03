package simulation;

/**
 * Represents a particle in the N-body simulation. Positions and velocities are
 * updated in-place by the integrator.
 */
public class Particle {
    private final int id;
    private final double mass;
    private Vector3 position;
    private Vector3 velocity;

    public Particle(int id, double mass, Vector3 position, Vector3 velocity) {
        this.id = id;
        this.mass = mass;
        this.position = position;
        this.velocity = velocity;
    }

    public Particle copy() {
        return new Particle(id, mass, position, velocity);
    }

    public int getId() {
        return id;
    }

    public double getMass() {
        return mass;
    }

    public Vector3 getPosition() {
        return position;
    }

    public void setPosition(Vector3 position) {
        this.position = position;
    }

    public Vector3 getVelocity() {
        return velocity;
    }

    public void setVelocity(Vector3 velocity) {
        this.velocity = velocity;
    }
}
