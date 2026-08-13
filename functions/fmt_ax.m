function fmt_ax()
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 20, ...
             'FontWeight', 'bold', 'LineWidth', 1.0, 'TickLabelInterpreter', 'tex');
             
    set(get(gca,'XLabel'), 'FontName', 'Times New Roman', 'FontSize', 20, ...
                           'FontWeight', 'bold', 'Interpreter', 'tex');
    set(get(gca,'YLabel'), 'FontName', 'Times New Roman', 'FontSize', 20, ...
                           'FontWeight', 'bold', 'Interpreter', 'tex');
                           
    lgd = findobj(gca,'Type','Legend');
    if ~isempty(lgd)
        lgd.FontName = 'Times New Roman';
        lgd.FontSize = 20; 
        lgd.FontWeight = 'bold';
        lgd.Interpreter = 'tex';
    end
end