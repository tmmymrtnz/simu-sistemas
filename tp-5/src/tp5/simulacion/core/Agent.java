package tp5.simulacion.core;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Represents a single pedestrian. Central (fixed) obstacle can be represented by setting movable=false.
 */
public class Agent {
    private static final AtomicInteger ID_GENERATOR = new AtomicInteger(0);

    private final int id;
    private final boolean movable;

    private double radius;
    private double mass;

    private Vector2 position;
    private Vector2 velocity;
    private Vector2 desiredDirection;
    private double desiredSpeed;
    private Vector2 acceleration;
    private Vector2 heading;

    public Agent(
            boolean movable,
            double radius,
            double mass,
            Vector2 position,
            Vector2 velocity,
            Vector2 desiredDirection,
            double desiredSpeed
    ) {
        this.id = ID_GENERATOR.getAndIncrement();
        this.movable = movable;
        this.radius = radius;
        this.mass = mass;
        this.position = position;
        this.velocity = velocity;
        this.desiredDirection = desiredDirection.normalized();
        this.desiredSpeed = desiredSpeed;
        this.acceleration = Vector2.ZERO;
        this.heading = this.desiredDirection;
    }

    public int getId() {
        return id;
    }

    public boolean isMovable() {
        return movable;
    }

    public double getRadius() {
        return radius;
    }

    public void setRadius(double radius) {
        this.radius = radius;
    }

    public double getMass() {
        return mass;
    }

    public void setMass(double mass) {
        this.mass = mass;
    }

    public Vector2 getPosition() {
        return position;
    }

    public void setPosition(Vector2 position) {
        this.position = position;
    }

    public Vector2 getVelocity() {
        return velocity;
    }

    public void setVelocity(Vector2 velocity) {
        this.velocity = velocity;
    }

    public Vector2 getDesiredDirection() {
        return desiredDirection;
    }

    public void setDesiredDirection(Vector2 desiredDirection) {
        this.desiredDirection = desiredDirection.normalized();
    }

    public double getDesiredSpeed() {
        return desiredSpeed;
    }

    public void setDesiredSpeed(double desiredSpeed) {
        this.desiredSpeed = desiredSpeed;
    }

    public Vector2 getAcceleration() {
        return acceleration;
    }

    public void setAcceleration(Vector2 acceleration) {
        this.acceleration = acceleration;
    }

    public Vector2 getHeading() {
        if (heading == null || heading.normSq() == 0) {
            return desiredDirection;
        }
        return heading;
    }

    public void setHeading(Vector2 heading) {
        if (heading == null || heading.normSq() == 0) {
            this.heading = desiredDirection;
        } else {
            this.heading = heading.normalized();
        }
    }
}
