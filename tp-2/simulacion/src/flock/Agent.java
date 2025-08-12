package flock;

public class Agent implements Grid.RadiusProvider {
    public final int id;
    public double x, y;
    public double theta;       // ángulo actual
    public double nextTheta;   // buffer de actualización
    public final double v0;    // módulo de velocidad
    public double r;           // radio (0 = puntual)

    public Agent(int id, double x, double y, double theta, double v0){
        this(id, x, y, theta, v0, 0.0);
    }

    public Agent(int id, double x, double y, double theta, double v0, double r){
        this.id = id;
        this.x = x;
        this.y = y;
        this.theta = theta;
        this.v0 = v0;
        this.r = Math.max(0.0, r);
    }

    public double vx(){ return v0 * Math.cos(theta); }
    public double vy(){ return v0 * Math.sin(theta); }

    @Override
    public double radius(){ return r; }
}
