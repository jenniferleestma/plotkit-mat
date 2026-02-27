function Avg_with_1SD_Haze(xaxis, mean, stdev, color, lineStyle)

% PLOT AVG LINE WITH HAZE FOR CONFIDENCE INTERVAL
if isempty(color)
    color = 'k';
end
if isempty(lineStyle)
    lineStyle = '-';
end

% Set up plot parameters
lineWidth = 3;

% Plot figure
hold on;
plot(xaxis, mean, 'Color', color, 'LineWidth', lineWidth, 'LineStyle', lineStyle); % mean line
xData2 = [xaxis, fliplr(xaxis)]; % not important, step to make visualization
inBetween = [(mean+stdev)', fliplr((mean-stdev)')];
fill(xData2, inBetween, color,'FaceAlpha',0.3,'EdgeAlpha',0);
end