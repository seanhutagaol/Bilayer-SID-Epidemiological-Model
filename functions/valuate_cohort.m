function [t, a_ps, a_pi, A_pi, A_pd] = valuate_cohort(tau0, T, dt, p, fp, t_mac, I_eff_mac, y0, r)
    t = (0:dt:T)';
    [~, y] = ode45(@(tt,yy) ode_cohort(tt,yy,p,fp,t_mac,I_eff_mac,tau0), t, y0);

    ps = max(0, y(:,1));
    pi_ = max(0, y(:,2));

    delta   = p(3);
    k_      = fp(4);
    mu_     = fp(2);
    kdelta  = k_ * delta;

    disc = exp(-r*t);
    a_ps = cumtrapz(t, ps  .* disc);
    a_pi = cumtrapz(t, pi_ .* disc);

    A_pi = disc .* pi_ + (r + mu_ + kdelta) .* a_pi;
    A_pd = kdelta .* a_pi;
end