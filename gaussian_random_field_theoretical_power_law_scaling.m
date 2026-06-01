syms y1 y2 y3
x1 = 1/sqrt(3)*y1 - 1/sqrt(2)*y2 - 1/sqrt(6)*y3;
x2 = 1/sqrt(3)*y1 + 1/sqrt(2)*y2 - 1/sqrt(6)*y3;
x3 = 1/sqrt(3)*y1 + sqrt(2)/sqrt(3)*y3;

simplify(-1/2*(x1^2+x2^2+x3^2) + 1/10*(x1+x2+x3)^2)
%%
original_expression = simplify(simplify((x2-x1) * (x3-x1) * (x3-x2)) * simplify(-x1*x2*x3));

% expression = (3^(1/2)*y2^3*y3^3)/3  + (2^(1/2)*6^(1/2)*y2^3*y3^3)/9 ...
%     - (3^(1/2)*y2^5*y3)/6 - (5*3^(1/2)*y2*y3^5)/54 - (2^(1/2)*6^(1/2)*y2*y3^5)/27 ...
%     - (2^(1/2)*3^(1/2)*y1*y2^5)/12 ...
%     + (4*6^(1/2)*y1*y2*y3^4)/27 + (11*2^(1/2)*3^(1/2)*y1*y2*y3^4)/108 ...
%     + (2^(1/2)*3^(1/2)*y1^3*y2^3)/18 ...
%     - (2*6^(1/2)*y1^3*y2*y3^2)/27 - (5*2^(1/2)*3^(1/2)*y1^3*y2*y3^2)/54   ...
%     + (2^(1/2)*3^(1/2)*y1*y2^3*y3^2)/6;
% expression = simplify(expression);

% monomial = 5*sqrt(3)/9*v^3*w^3 - sqrt(3)/6*v^5*w - 25*sqrt(3)/54*v*w^5 - ...
%     sqrt(6)/12*u*v^5 + 27*sqrt(6)/108*u*v*w^4 + sqrt(6)/18*u^3*v^3 - ...
%     sqrt(6)/6*u^3*v*w^2 + sqrt(6)/6*u*v^3*w^2;
expression = exp(-1/5*y1^2)*exp(-1/2*y2^2)*exp(-1/2*y3^2) * original_expression;

first_int = simplify(int(expression, y1, -inf, -sqrt(2)*y3));
second_int = simplify(int(first_int, y2, 0, sqrt(3)*y3));
third_int = simplify(int(second_int, y3, 0, inf));

pretty(simplify(third_int * sqrt(2)/4/sqrt(5)/pi^(5/2)))
double(third_int * sqrt(2)/4/sqrt(5)/pi^(5/2))

%%
syms y1 y2 y3

r1 = 1/sqrt(3)*y1 - 1/sqrt(2)*y2 - 1/sqrt(6)*y3;
r2 = 1/sqrt(3)*y1 + 1/sqrt(2)*y2 - 1/sqrt(6)*y3;
r3 = 1/sqrt(3)*y1 + sqrt(2)/sqrt(3)*y3;

original_expression = simplify(simplify((r2-r1) * (r3-r1) * (r3-r2)) * simplify(-r1*r2*r3));
exponent = exp(simplify(-1/2*(r1^2+r2^2+r3^2) + 1/10*(r1+r2+r3)^2));

first_int = simplify(int(exponent*original_expression, y1, -inf, -sqrt(2)*y3));
second_int = simplify(int(first_int, y2, 0, sqrt(3)*y3));
third_int = simplify(int(second_int, y3, 0, inf));

pretty(simplify(third_int * sqrt(2)/4/sqrt(5)/pi^(5/2)))
double(third_int * sqrt(2)/4/sqrt(5)/pi^(5/2))