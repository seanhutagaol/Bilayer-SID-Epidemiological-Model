function save_fig(f, fname)
    exportgraphics(f, [fname, '.jpeg'], 'Resolution', 300);
    fprintf('  Saved: %s.jpeg\n', fname);
end