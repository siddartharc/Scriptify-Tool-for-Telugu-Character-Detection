import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
def Clustering(results):
    # Process the detected objects
    # boxes_data = results.boxes.data.tolist()

    # Sort boxes based on x-axis
    sorted_boxes_data = sorted(results, key=lambda x: x[0])  # Sort by x-axis
    boxes_coords = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, score, class_id in sorted_boxes_data])
    while True:
        # step_size = int(input("Enter the step size based on you writing : "))
        clusters = []
        for i in range(len(boxes_coords)):
            clusters.append([i])
        # print(clusters)
        step_size = int(input("enter the step size based on your handwriting ranging from 10-200 : "))
        # Define reference lines and create sub-clusters within each main cluster
        reference_lines = list(range(0, 2000, step_size))  # Adjust range and step as needed
        line_clusters = [[] for _ in range(len(reference_lines) - 1)]
        for cluster in clusters:
            for point_idx in cluster:
                y_coord = boxes_coords[point_idx, 1]
                for idx, line in enumerate(reference_lines[:-1]):
                    if line <= y_coord <= reference_lines[idx + 1]:
                        line_clusters[idx].append(point_idx)
                        break

        # Define colors for clusters
        colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        print('check if all the points are correctly clusterd')
        # Plot the points on a graph with different colors for sub-clusters
        for line_idx, sub_cluster in enumerate(line_clusters):
            color_idx = line_idx % len(colors)
            for point_idx in sub_cluster:
                x, y = boxes_coords[point_idx, 0], boxes_coords[point_idx, 1]
                plt.scatter(x, y, color=colors[color_idx])  # Plot the point with cluster-specific color

        # Display the graph
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Object Detection Points with Sub-Clusters')
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.grid(True)
        plt.show()

        user_input = input("Are the lines correctly Clusterd ? (Y/N): ")
        if user_input.lower() == 'y':
            return line_clusters,boxes_coords
            break