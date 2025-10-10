function fx = f_osc_nl(x, theta)
% Nonlinear damped oscillator flow
% theta = [k; c; a]
k = theta(1);
c = theta(2);
a = theta(3);
x1 = x(1); x2 = x(2);

fx = zeros(2,1);
fx(1) = x2;
fx(2) = -k*x1 - c*x2 + a*tanh(x1);
end
