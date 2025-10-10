function y = g_lin(x, ~)
% Linear observation of first state
% Accepts either x as vector (nx×1) or matrix (nx×T)
if isvector(x); x = x(:); end
y = x(1,:).';   % return column vector (ny=1)
end
