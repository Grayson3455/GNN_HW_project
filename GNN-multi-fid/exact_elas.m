clc
clear 
close all

n  = 1000;

R = 0.09; %outer radius
r = 0.075; %inner radius
theta = 0:2*pi/n:2*pi;
r_outer = ones(size(theta))*R;
r_inner = ones(size(theta))*r;
rr = [r_outer;r_inner];
theta = [theta;theta];
xx = rr.*cos(theta);
yy = rr.*sin(theta);
zz = zeros(size(xx));

E  = 200e9;
nu = 0.3;
P  = 30e6;

rr = sqrt(xx.^2 + yy.^2);

zz_exact = 1e3* P * (1+nu)/E * (rr / ( (R/r)^2-1 ) ) .* (1-2*nu + (R./rr).^2);

surf(xx,yy,zz_exact)
view(2)
shading interp
axis equal
colorbar