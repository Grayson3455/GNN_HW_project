//+
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.075, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 0.09, 0, 2*Pi};
//+
Curve Loop(1) = {2};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};
