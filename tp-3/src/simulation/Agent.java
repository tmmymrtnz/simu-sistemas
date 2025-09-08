package simulation;

public class Agent {
    public final int id;
    public double x, y;
    public double vx, vy;
    public double r;
    public final double mass = 1.0;

    public Agent(int id, double x, double y, double vx, double vy){
        this(id, x, y, vx, vy, 0.0015);
    }

    public Agent(int id, double x, double y, double vx, double vy, double r){
        this.id = id;
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.r = r;
    }

    public double vx(){ return vx; }
    public double vy(){ return vy; }
    public double radius(){ return r; }
}
