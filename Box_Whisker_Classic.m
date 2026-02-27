function Box_Whisker_Classic(x,data, width, edgeColor, faceColor, meanMarkerSize, addPoints)
%BOX N WHISKER Because the default one is terrible
%   This will plot a box and whisker plot with optional data points
%   included for each bin.  This is meant to plot the data for one box!  So
%   you just need to call this function with each set of data, along with
%   the x-position you want this plotted on.

%   Created by: Jenny Leestma (01/2025)

%   INPUTS:
%   x: the position on the x-axis of the box
%   data: the data that should make up the box
%   width: width of each box, this will change depending on the width of
%   your graph and number of boxes, so this is left as a tunable variable.
%   color: self explanatory
%   addPoints: boolean (true or false), will add individual data points
%   behind bars

    mLow = min(data);
    mHigh = max(data);
    med = median(data);
    qLow = quantile(data, 0.25); % First quartile (25th percentile)
    qHigh = quantile(data, 0.75); % Third quartile (75th percentile)
    m = mean(data);


    % Define rectangle vertices
    xSquare = [x-width/2 x+width/2 x+width/2 x-width/2]; % X-coordinates
    ySquare = [qLow qLow qHigh qHigh]; % Y-coordinates
    
    % Plot the filled rectangle
    % fill(xSquare, ySquare, color, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    fill(xSquare, ySquare, faceColor, 'EdgeColor', edgeColor, 'LineWidth', 1);
    % Adjust axis limits

    % Plot individual points - in front of box
    if addPoints
        jitterX = (rand(length(data),1))*width + x - width/2; % this will make the jitter as wide as the boxes
        plot(jitterX, data, 'o','MarkerEdgeColor','k','MarkerFaceColor','none','MarkerSize',3,'LineWidth',1)
    end

    % PLOT LINES
    plot([x,x], [qHigh, mHigh], '-','Color','k','LineWidth',1.5)
    plot([x-width/4,x+width/4], [mHigh, mHigh], '-','Color','k','LineWidth',1.5)
    plot([x,x], [qLow, mLow], '-','Color','k','LineWidth',1.5)
    plot([x-width/4,x+width/4], [mLow, mLow], '-','Color','k','LineWidth',1.5)
    plot([x-width/2 x+width/2], [med,med], '-','Color','k','LineWidth',1.5)
    plot(x, m, 's','MarkerFaceColor','k', 'MarkerEdgeColor', 'k','MarkerSize',meanMarkerSize)

end