function fill_band(x, lo, hi)
    fill([x; flipud(x)], [lo; flipud(hi)], ...
         [0.82 0.82 0.82], 'EdgeColor','none','HandleVisibility','off');
end