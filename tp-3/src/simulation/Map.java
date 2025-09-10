package simulation;

import java.util.ArrayList;
import java.util.List;

public class Map {

    public final double L_fixed = 0.09;
    public final double L;
    public final List<Wall> walls = new ArrayList<>();
    public final List<Agent> particles = new ArrayList<>();

    public Map(double L){
        this.L = L;
        // Caja izquierda
        walls.add(new Wall(0, 0, L_fixed, 0));
        walls.add(new Wall(0, 0, 0, L_fixed));
        walls.add(new Wall(0, L_fixed, L_fixed, L_fixed));

        // Dos “puertas” verticales
        if (L != L_fixed) { // evitar solapamiento paredes si L = L_fixed
            walls.add(new Wall(L_fixed, 0, L_fixed, (L_fixed - L)/2));
            walls.add(new Wall(L_fixed, L_fixed, L_fixed, (L_fixed + L)/2));
        }

        // Pasillo horizontal
        walls.add(new Wall(L_fixed, (L_fixed - L)/2, 2*L_fixed, (L_fixed - L)/2));
        walls.add(new Wall(L_fixed, (L_fixed + L)/2, 2*L_fixed, (L_fixed + L)/2));

        // Pared derecha
        walls.add(new Wall(2*L_fixed, (L_fixed + L)/2, 2*L_fixed, (L_fixed - L)/2));
    }

    /** Crea qty partículas con velocidad de módulo fijo y direcciones aleatorias. */
    public List<Agent> createParticles(int qty){
        List<Agent> newParticles = new ArrayList<>();
        double speed = 0.01;
        double r = 0.0015;
        double min_dist_sq = Math.pow(2 * r, 2);

        int baseId = this.particles.size();
        for (int i = 0; i < qty; i++) {
            double x, y;
            int attempts = 0;
            final int max_attempts = 10000; // Avoid infinite loops

            while (true) {
                attempts++;
                if (attempts > max_attempts) {
                    throw new RuntimeException("Could not place particle " + (baseId + i) +
                                               " without overlap after " + max_attempts + " attempts.");
                }

                boolean overlaps = false;
                // Dentro del cuadrado [r, L_fixed - r]
                x = Math.random() * (L_fixed - 2*r) + r;
                y = Math.random() * (L_fixed - 2*r) + r;

                // Check against particles from previous batches
                for (Agent p : this.particles) {
                    double dx = x - p.x;
                    double dy = y - p.y;
                    if (dx*dx + dy*dy < min_dist_sq) {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps) continue;

                // Check against new particles from this batch
                for (Agent p : newParticles) {
                    double dx = x - p.x;
                    double dy = y - p.y;
                    if (dx*dx + dy*dy < min_dist_sq) {
                        overlaps = true;
                        break;
                    }
                }

                if (!overlaps) {
                    break; // Found a valid position
                }
            }

            double angle = Math.random() * 2 * Math.PI;
            double vx = speed * Math.cos(angle);
            double vy = speed * Math.sin(angle);

            Agent particle = new Agent(baseId + i, x, y, vx, vy, r);
            newParticles.add(particle);
        }
        this.particles.addAll(newParticles);
        return newParticles;
    }

    /** Dibujo ASCII opcional. */
    public void printMap() {
        int size = (int) Math.round(2 * L_fixed / 0.01) + 2;
        char[][] grid = new char[size][size];
        for (int y = 0; y < size; y++) for (int x = 0; x < size; x++) grid[y][x] = ' ';
        for (Wall wall : walls) {
            int x1 = (int) Math.round(wall.x1 / 0.01);
            int y1 = (int) Math.round(wall.y1 / 0.01);
            int x2 = (int) Math.round(wall.x2 / 0.01);
            int y2 = (int) Math.round(wall.y2 / 0.01);
            if (wall.isHorizontal()) {
                for (int x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) grid[y1][x] = '-';
            } else {
                for (int y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) grid[y][x1] = '|';
            }
        }
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) System.out.print(grid[y][x]);
            System.out.println();
        }
    }
}
