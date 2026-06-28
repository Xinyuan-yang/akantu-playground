// use multiples of 10
Nb_elems = 200;

// Mesh.Format = 2.2;

x_size = 0.5;
y_size = 0.1;

h_def = x_size / Nb_elems;
h_bot = h_def;

Point(1) = {0, 0, 0, h_def};
Point(2) = {x_size, 0, 0, h_def};
Point(3) = {x_size, y_size, 0, h_def};
Point(4) = {0, y_size, 0, h_def};

Point(5) = {0, -y_size, 0, h_def};
Point(6) = {x_size, -y_size, 0, h_def};
Point(7) = {x_size, 0, 0, h_def};
Point(8) = {0, 0, 0, h_def};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Physical Point("base_bottom_left") = {5};
Physical Point("base_bottom_right") = {6};

Physical Line("slider_top") = {3};
Physical Line("slider_bottom") = {1};
Physical Line("slider_left") = {4};
Physical Line("slider_right") = {2};

Physical Line("base_bottom") = {5};
Physical Line("base_top") = {7};
Physical Line("base_left") = {8};
Physical Line("base_right") = {6};

Physical Surface("slider") = {1};
Physical Surface("base") = {2};

Transfinite Surface "*";
Recombine Surface "*";

Mesh.SecondOrderIncomplete = 1;
