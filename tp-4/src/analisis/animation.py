import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import os


def parse_xyz_file(filepath):
    """
    Parses a custom .txt file with XYZ-like format for particle data.
    """
    if not os.path.exists(filepath):
        print(f"File not found at: {filepath}")
        return []
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    particles_data = []
    i = 0
    while i < len(lines):
        try:
            current_line = lines[i].strip()
            
            if not current_line or current_line.startswith("#"):
                i += 1
                continue
            
            num_particles = int(current_line) 

            time_str = lines[i + 1].strip()
            time = float(time_str.split('=')[1])
            
            frame_data = {'time': time, 'positions': [], 'velocities': [], 'num_particles': num_particles} 
            
            for j in range(num_particles):
                line = lines[i + 2 + j].strip().split()
                pos = [float(x) for x in line[1:4]]
                vel = [float(x) for x in line[4:7]]
                
                frame_data['positions'].append(pos)
                frame_data['velocities'].append(vel)
            
            particles_data.append(frame_data)
            i += 2 + num_particles 
            
        except (ValueError, IndexError):
            break
            
    return particles_data


def update_plot(frame, particles_data, scatter1, scatter2, cluster_size):
    """
    Updates the scatter plot(s) for each animation frame.
    """
    frame_data = particles_data[frame]
    positions = np.array(frame_data['positions'])

    scatter1._offsets3d = (
        positions[:cluster_size, 0],
        positions[:cluster_size, 1],
        positions[:cluster_size, 2]
    )

    return_objects = [scatter1]

    if scatter2 is not None:
        scatter2._offsets3d = (
            positions[cluster_size:, 0],
            positions[cluster_size:, 1],
            positions[cluster_size:, 2]
        )
        return_objects.append(scatter2)

    global ax
    ax.set_title(f"Time: {frame_data['time']:.2f}")

    return return_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate N-body simulation from a data file.")
    parser.add_argument("file", help="Path to the PARTICLE DATA file (e.g., cluster_200_run000_particles.txt).")
    parser.add_argument("--output", help="Output filename for the animation (e.g., animation.mp4).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation.")
    parser.add_argument("--cluster-size", type=int, default=100, help="Number of particles per cluster. If total N = this size, only one group is plotted.")
    
    args = parser.parse_args()

    input_file_path = args.file

    print(f"Reading data from {input_file_path}...")
    particles_data = parse_xyz_file(input_file_path)
    
    if not particles_data:
        print("No particle data found. Ensure the file contains particle position blocks.")
        print("Exiting.")
        exit(1)

    total_particles = particles_data[0]['num_particles'] 
    cluster_size = args.cluster_size
    
    # Check if there should be a second cluster: total particles > size of the first cluster
    has_second_cluster = total_particles > cluster_size

    fig = plt.figure(figsize=(8, 8))
    global ax
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    initial_positions = np.array(particles_data[0]['positions'])

    ax.set_xlim([initial_positions[:, 0].min() - 1, initial_positions[:, 0].max() + 1])
    ax.set_ylim([initial_positions[:, 1].min() - 1, initial_positions[:, 1].max() + 1])
    ax.set_zlim([initial_positions[:, 2].min() - 1, initial_positions[:, 2].max() + 1])

    cluster1_pos = initial_positions[:cluster_size]
    
    color1 = 'blue' if has_second_cluster else 'green' 
    scatter1 = ax.scatter(cluster1_pos[:, 0], cluster1_pos[:, 1], cluster1_pos[:, 2],
                          s=50, color=color1, label='Cluster 1' if has_second_cluster else 'All Particles')
    
    scatter2 = None 
    
    if has_second_cluster:
        cluster2_pos = initial_positions[cluster_size:]
        scatter2 = ax.scatter(cluster2_pos[:, 0], cluster2_pos[:, 1], cluster2_pos[:, 2],
                              s=50, color="#FF00C8", label='Cluster 2')

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=len(particles_data),
        fargs=(particles_data, scatter1, scatter2, cluster_size), 
        interval=1000 / args.fps,
        blit=False,
        repeat=False
    )

    if args.output:
        print(f"Saving animation to {args.output}...")
        ani.save(args.output, writer='ffmpeg', fps=args.fps)
    else:
        plt.show()