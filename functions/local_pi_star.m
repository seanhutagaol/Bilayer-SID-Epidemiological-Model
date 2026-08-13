function pi_star = local_pi_star(scenario, tau0, T, dt, p, fp, t_mac_base, x0_mac, y0, r)
    [~, x_m] = ode45(@(t,y) ode_macro(t,y,p,fp), t_mac_base, x0_mac);
    I_eff = max(0, x_m(:,3) + fp(5) * x_m(:,4));

    [t, a_ps, a_pi, A_pi, A_pd] = valuate_cohort(tau0, T, dt, p, fp, t_mac_base, I_eff, y0, r);

    switch scenario
        case 3
            B = a_pi + A_pd;
        case 4
            B = A_pi + A_pd;
        otherwise
            error('local_pi_star: scenario must be 3 or 4');
    end

    psi = NaN(size(t));
    valid = t > 0;
    psi(valid) = B(valid) ./ a_ps(valid);

    pi_star = max(psi);
end