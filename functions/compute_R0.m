function R0 = compute_R0(beta, delta, Lambda, nu, k, gamma_art)
    s_str  = Lambda / (Lambda + nu + eps);
    sp_str = nu     / (Lambda + nu + eps);
    R0 = (beta * s_str) / (Lambda + delta) + ...
         (beta * gamma_art * sp_str) / (Lambda + k * delta);
end