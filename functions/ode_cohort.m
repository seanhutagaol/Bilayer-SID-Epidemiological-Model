function dydt = ode_cohort(t, y, p, fp, t_mac, I_eff_mac, tau0)
    ps  = y(1);  pi_ = y(2);

    beta  = p(1);  alpha = p(2);  delta = p(3);
    mu    = fp(2);
    k     = fp(4);
    delta_p = k * delta;

    I_env = interp1(t_mac, I_eff_mac, t + tau0, 'linear', 'extrap');
    I_env = max(0, I_env);

    lam = beta * exp(-alpha * I_env) * I_env;

    dps = -(lam + mu) * ps;
    dpi =  lam * ps - (mu + delta_p) * pi_;
    dpd =  delta_p * pi_;

    dydt = [dps; dpi; dpd];
end