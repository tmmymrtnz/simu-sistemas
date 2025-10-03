import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import os

# --- PATH BUILDING VARIABLES (Removed - we use args.file instead) ---
# NOTE: Removed the hardcoded path logic (script_dir, base_dir, file_path) 
# to ensure the script uses the file path passed via the Makefile (args.file).
# This prevents conflict and respects user input.

def parse_xyz_file(filepath):
    """
    Parses a custom .txt file with XYZ-like format for particle data.
    The format is:
    N
    t=<time>
    mass pos_x pos_y pos_z vel_x vel_y vel_z
    ...
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

            time_str = lines[i+1].strip()
            time = float(time_str.split('=')[1])
            
            frame_data = {'time': time, 'positions': [], 'velocities': []}
            
            for j in range(num_particles):
                line = lines[i+2+j].strip().split()
                pos = [float(x) for x in line[1:4]]
                vel = [float(x) for x in line[4:7]]
                
                frame_data['positions'].append(pos)
                frame_data['velocities'].append(vel)
            
            particles_data.append(frame_data)
            i += 2 + num_particles 
            
        except (ValueError, IndexError) as e:
            # This block means either the file ended, or the format was wrong (e.g., trying to read 
            # the time line or particle data line as N). We break because we hit unexpected data.
            # print(f"Debug: Failed to parse line {i+1}. Error: {e}") 
            break
            
    return particles_data

def update_plot(frame, particles_data, scatter):
    """
    Updates the scatter plot for each animation frame.
    """
    frame_data = particles_data[frame]
    positions = np.array(frame_data['positions'])
    
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    
    global ax 
    ax.set_title(f"Time: {frame_data['time']:.2f}")

    return scatter,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate N-body simulation from a data file.")
    # NOTE: The 'file' argument now correctly describes the intended input.
    parser.add_argument("file", help="Path to the PARTICLE DATA file (e.g., cluster_200_run000_particles.txt).")
    parser.add_argument("--output", help="Output filename for the animation (e.g., animation.mp4).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation.")
    
    args = parser.parse_args()

    input_file_path = args.file

    print(f"Reading data from {input_file_path}...")
    particles_data = parse_xyz_file(input_file_path)
    
    if not particles_data:
        print("No particle data found. Ensure the file contains particle position blocks.")
        print("Exiting.")
        exit(1)

    # ... (Rest of the animation setup)
    
    fig = plt.figure(figsize=(8, 8))
    # 'ax' needs to be globally accessible for 'update_plot' to work correctly
    global ax
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    initial_positions = np.array(particles_data[0]['positions'])
    ax.set_xlim([initial_positions[:, 0].min() - 1, initial_positions[:, 0].max() + 1])
    ax.set_ylim([initial_positions[:, 1].min() - 1, initial_positions[:, 1].max() + 1])
    ax.set_zlim([initial_positions[:, 2].min() - 1, initial_positions[:, 2].max() + 1])
    
    scatter = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], s=50)

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=len(particles_data),
        fargs=(particles_data, scatter),
        interval=1000 / args.fps,
        blit=False,
        repeat=False
    )

    if args.output:
        print(f"Saving animation to {args.output}...")
        ani.save(args.output, writer='ffmpeg', fps=args.fps)
    else:
        plt.show()