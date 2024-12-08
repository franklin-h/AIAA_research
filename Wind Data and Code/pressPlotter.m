% MATLAB Script for Point Cloud Visualization and Mesh Generation

% Step 1: Read the point cloud data from the CSV file
filename = 'pressure_point_cloud_2.csv'; % Replace with the actual file name
data = readmatrix(filename);

% Assuming the file contains columns for x, y, z coordinates
x = data(:, 1);
y = data(:, 2);
z = data(:, 3);
pressure = data(:,4); 
% Step 2: Display the point cloud
fig1 = figure;
scatter3(x,y,z,10,pressure,'filled'); % Colored by z-axis values
colormap('jet')
[ts,ss] = title('Payload Surface Pressure During Descent',' ',Interpreter='latex');
ylabel('Size (meters)',Interpreter='latex');
% ylabel('Y',Interpreter='latex');
% zlabel('Z',Interpreter='latex');

a=colorbar;
a.FontSize = 20; 
a.Label.FontSize = 20; 

a.Label.String = 'Pressure (Pa)';
a.Label.Interpreter = 'latex'; 
a.TickLabelInterpreter = 'latex'; 
view(315, 45);
camup([0 1 0])
ax = gca; 
ax.FontSize = 20;
ax.TickLabelInterpreter = 'latex'; 
% grid(ax,'off')
% view([0 -20 90])


axis equal;
grid on;

exportgraphics(fig1,"pressure_contour.pdf",Resolution=300)
% Step 3: Create a 3D mesh using Delaunay triangulation
points = [x, y, z];
tri = delaunay(x, y); % Delaunay triangulation based on x and y coordinates

% % Step 4: Visualize the 3D mesh
% figure;
% trisurf(tri, x, y, z, 'FaceColor', 'cyan', 'EdgeColor', 'none');
% title('3D Mesh from Point Cloud');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% axis equal;
% grid on;
% lighting gouraud;
% camlight headlight;
