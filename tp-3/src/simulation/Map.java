package simulation;

import java.util.ArrayList;
import java.util.List;

public class Map{
    
    public final double L_fixed = 0.09;
    public final double L;
    public final List<Wall> walls = new ArrayList<>();
    public final List<Agent> particles = new ArrayList<>();


    public Map(double L){
        this.L = L;
        walls.add(new Wall(0, 0, L_fixed, 0));
        walls.add(new Wall(0, 0, 0, L_fixed));
        walls.add(new Wall(0,L_fixed,L_fixed,L_fixed));

        walls.add(new Wall(L_fixed,0,L_fixed,(L_fixed-L)/2));
        walls.add(new Wall(L_fixed,L_fixed,L_fixed,(L_fixed+L)/2));

        walls.add(new Wall(L_fixed,(L_fixed-L)/2, 2* L_fixed, (L_fixed-L)/2));
        walls.add(new Wall(L_fixed,(L_fixed+L)/2, 2*L_fixed,(L_fixed+L)/2));

        walls.add(new Wall(2*L_fixed,(L_fixed+L)/2,2* L_fixed, (L_fixed-L)/2));

    }

    public List<Agent> createParticles(int qty){
        List<Agent> newParticles = new ArrayList<>();
        for (int i = 0; i < qty; i++) {
            double x = Math.random() * (L_fixed);
            double y = Math.random() * (L_fixed);
            double angle = Math.random() * 2 * Math.PI;
            double speed = 0.01; 
            double vx = speed * Math.cos(angle);
            double vy = speed * Math.sin(angle);
            Agent particle = new Agent(x, y, vx, vy);
            newParticles.add(particle);
        }
        this.particles.addAll(newParticles);
        return newParticles;
    }

    public void printMap() {
        int size = (int) Math.round(2 * L_fixed / 0.01) + 2; // Ajusta la resolución aquí
        char[][] grid = new char[size][size];

        // Inicializa el grid con espacios
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                grid[y][x] = ' ';
            }
        }

        // Dibuja las paredes
        for (Wall wall : walls) {
            int x1 = (int) Math.round(wall.x1 / 0.01);
            int y1 = (int) Math.round(wall.y1 / 0.01);
            int x2 = (int) Math.round(wall.x2 / 0.01);
            int y2 = (int) Math.round(wall.y2 / 0.01);

            if (wall.isHorizontal()) {
                for (int x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
                    grid[y1][x] = '-';
                }
            } else {
                for (int y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
                    grid[y][x1] = '|';
                }
            }
        }

        // Imprime el grid
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                System.out.print(grid[y][x]);
            }
            System.out.println();
        }
    }

}