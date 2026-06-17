Point(1) = {0, 1.1, 0, 0.1};
Point(2) = {12.0, 1.1, 0, 0.1};
Point(3) = {12.0, 7.0, 0, 0.1};
Point(4) = {0.0, 7.0, 0, 0.1};

Point(5) = {16.0, 1.0, 0, 0.2};
Point(6) = {16.0, 0.0, 0, 0.2};
Point(7) = {-2.0, 0.0, 0, 0.2};
Point(8) = {-2.0, 1.0, 0, 0.2};

// Define lines with midpoints, each with transfinite meshing
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Line(5) = {8, 7};
Line(6) = {7, 6};
Line(7) = {6, 5};
Line(8) = {5, 8};

Line Loop(11) = {5, 6, 7, 8};
Plane Surface(11) = {11};

Physical Surface(2) = {1};
Physical Line("contact_top") = {1};

Physical Surface(12) = {11};
Physical Line("contact_bottom") = {8};

Physical Line("loading") = {3};

Physical Line("YFixed") = {6};
Physical Line("XFixed") = {6};
