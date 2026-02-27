function h = Custom_Violin(data, xpos, varargin)
% HALVIVIOLIN  Plots a half or full violin plot with median and quartiles
%
%   h = halfViolin(data, xpos)
%   Optional name-value pairs:
%       'Width'  : maximum half-width of violin (default 0.3)
%       'Color'  : fill color (default [0.5 0.5 0.5])
%       'Alpha'  : transparency (default 0.3)
%       'Side'   : 'symmetric' (default), 'left', or 'right'

    % ---- Parse inputs ----
    p = inputParser;
    addParameter(p,'Width',0.3);
    addParameter(p,'Color',[0.5 0.5 0.5]);
    addParameter(p,'Alpha',1);
    addParameter(p,'Side','symmetric'); % 'symmetric' | 'left' | 'right'
    addParameter(p,'MeanMarkerSize',8)
    parse(p,varargin{:});

    width = p.Results.Width;
    color = p.Results.Color;
    alpha = p.Results.Alpha;
    side  = lower(p.Results.Side);
    meanMarkerSize = p.Results.MeanMarkerSize;

    % ---- Clean data ----
    data = data(:);
    data = data(~isnan(data));

    % ---- Kernel density estimate ----
    [f, yi] = ksdensity(data,'Function','pdf');
    f  = f(:);
    yi = yi(:);
    f = f ./ max(f) * width;

    % ---- Construct violin shape ----
    switch side
        case 'symmetric'
            x = [xpos + f; flipud(xpos - f)];
            y = [yi;       flipud(yi)];

        case 'right'
            x = [xpos; xpos + f; xpos];
            y = [yi(1); yi; yi(end)];

        case 'left'
            x = [xpos; xpos - f; xpos];
            y = [yi(1); yi; yi(end)];

        otherwise
            error('Side must be ''symmetric'', ''left'', or ''right''.')
    end

    % ---- Draw violin ----
    h = fill(x, y, color, 'FaceAlpha', 1, 'EdgeColor', color);
    hold on;

    % ---- Compute median and quartiles ----
    med   = median(data);
    qLow  = quantile(data, 0.25);
    qHigh = quantile(data, 0.75);
    m = mean(data);

    % ---- Helper to get width at a specific Y ----
    getWidthAtY = @(yval) interp1(yi, f, yval, 'linear', 0);

    medWidth  = getWidthAtY(med);
    qLowWidth = getWidthAtY(qLow);
    qHighWidth = getWidthAtY(qHigh);

    % ---- Draw median and quartile lines ----
    switch side
        case 'symmetric'
            plot([xpos-medWidth xpos+medWidth], [med med], 'k','LineWidth',1.5);
            plot([xpos-qLowWidth xpos+qLowWidth], [qLow qLow], 'k:','LineWidth',1.5);
            plot([xpos-qHighWidth xpos+qHighWidth], [qHigh qHigh], 'k:','LineWidth',1.5);
            plot(xpos, m, 'ks','MarkerSize',meanMarkerSize,'MarkerFaceColor','k');

        case 'right'
            plot([xpos xpos+medWidth], [med med], 'k','LineWidth',1.5);
            plot([xpos xpos+qLowWidth], [qLow qLow], 'k:','LineWidth',1.5);
            plot([xpos xpos+qHighWidth], [qHigh qHigh], 'k:','LineWidth',1.5);
            plot(xpos+medWidth/2, m, 'ks','MarkerSize',meanMarkerSize,'MarkerFaceColor','k');

        case 'left'
            plot([xpos-medWidth xpos], [med med], 'k','LineWidth',1.5);
            plot([xpos-qLowWidth xpos], [qLow qLow], 'k:','LineWidth',1.5);
            plot([xpos-qHighWidth xpos], [qHigh qHigh], 'k:','LineWidth',1.5);
            plot(xpos-medWidth/2, m, 'ks','MarkerSize',meanMarkerSize,'MarkerFaceColor','k');
    end
end