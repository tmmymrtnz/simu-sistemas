package simulation;

public class Wall {
    public double x1, x2, y1, y2;

    public Wall(double x1, double y1, double x2, double y2) {
        this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2;
    }

    public boolean isHorizontal() { return y1 == y2; }

    public double length() {
        return isHorizontal() ? Math.abs(x2 - x1) : Math.abs(y2 - y1);
    }
}
