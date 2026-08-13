function dydt = ode_macro(~, y, p, fp)
    s  = y(1);  sp = y(2);  i = y(3);  ip = y(4);

    beta   = p(1);  alpha = p(2);  delta = p(3);
    Lambda = fp(1);
    nu     = fp(3);
    k      = fp(4);
    gam    = fp(5);

    delta_p = k * delta;
    k_inf   = gam;

    I_eff = i + k_inf * ip;
    lam   = beta * exp(-alpha * I_eff) * I_eff;
    phi   = delta*i + delta_p*ip;

    ds  = Lambda*(1-s) - lam*s  - nu*s       + s*phi;
    dsp = nu*s         - lam*sp - Lambda*sp  + sp*phi;
    di  = lam*s  - (Lambda + delta  )*i  + i*phi;
    dip = lam*sp - (Lambda + delta_p)*ip + ip*phi;

    dydt = [ds; dsp; di; dip];
end