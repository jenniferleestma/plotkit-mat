function [p_val, R2, slope] = Get_Line_Stats(x,y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    coeffs = polyfit(x, y, 1);
    % Get fitted values
    fittedX = linspace(min(x), max(x), 200);
    fittedY = polyval(coeffs, fittedX);
    % Plot the fitted line
    mdl = fitlm(x,y);
    R2 = mdl.Rsquared.Ordinary;
    p_val = mdl.Coefficients.pValue;
    slope = mdl.Coefficients.Estimate(2);
end

