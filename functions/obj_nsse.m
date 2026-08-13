function err = obj_nsse(p, times, x0, fp, data)
    try
        [~, x] = ode45(@(t,y) ode_macro(t,y,p,fp), times, x0);
    catch
        err = 1e6; 
        return;
    end
    if any(isnan(x(:))) || any(x(:) < -1e-6)
        err = 1e6; 
        return;
    end
    k   = fp(4);
    dp  = k * p(3);
    s_p = x(:,3) + x(:,4);
    s_d = p(3).*x(:,3) + dp.*x(:,4);
    mp  = max(data(:,1));
    md  = max(data(:,2));
    if mp < eps || md < eps
        err = 1e6; 
        return; 
    end
    err = sum(((s_p - data(:,1))/mp).^2) + ...
          sum(((s_d - data(:,2))/md).^2);
end