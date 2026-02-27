function [comparison,means,h,gnames] = multcompare_n_adjust(stats,varargin)
%MULTCOMPARE Perform a multiple comparison of means or other estimates
%   MULTCOMPARE performs a multiple comparison using one-way anova or
%   anocova results to determine which estimates (such as means,
%   slopes, or intercepts) are significantly different.
%
%   COMPARISON = MULTCOMPARE(STATS) performs a multiple comparison
%   using a STATS structure that is obtained as output from any of
%   the following functions:  anova1, anova2, anovan, aoctool,
%   kruskalwallis, friedman.  The return value COMPARISON is a matrix
%   with one row per comparison and six columns.  Columns 1-2 are the
%   indices of the two samples being compared.  Columns 3-5 are a lower
%   bound, estimate, and upper bound for their difference. Column 6 is the
%   p-value for each individual comparison.
%
%   COMPARISON = MULTCOMPARE(STATS, 'PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%
%     'Alpha'       Specifies the confidence level as 100*(1-ALPHA)%
%                   (default 0.05).
%     'Approximate' Flag to indicate whether to compute the Dunnett
%                   critical value approximately for speed. Default is true
%                   for anovan and false otherwise.
%     'ControlGroup'   Index of the control group for Dunnett's test.
%                   Default is 1. Not valid for other CriticalValueType
%                   choices.
%     'CriticalValueType'  The type of critical value to use.  Choices are
%                   'tukey-kramer' (default), 'dunn-sidak', 'bonferroni',
%                   'scheffe', 'dunnett'.
%     'Dimension'   A vector specifying the dimension or dimensions over
%                   which the population marginal means are to be
%                   calculated.  Used only if STATS comes from anovan.
%                   The default is to compute over the first dimension
%                   associated with a categorical (non-continuous) factor.
%                   The value [1 3], for example, computes the population
%                   marginal mean for each combination of the first and
%                   third predictor values.
%     'Display'     Either 'on' (the default) to display a graph of the
%                   estimates with comparison intervals around them, or
%                   'off' to omit the graph.
%     'Estimate'    Estimate to compare.  Choices depend on the source of
%                   the stats structure:
%         anova1:  ignored, compare group means
%         anova2:  'column' (default) or 'row' means
%         anovan:  ignored, compare population marginal means
%         aoctool:  'slope', 'intercept', or 'pmm' (default is 'slope'
%                   for separate-slopes models, 'intercept' otherwise)
%         kruskalwallis:  ignored, compare average ranks of columns
%         friedman:  ignored, compare average ranks of columns
%
%   [COMPARISON,MEANS,H,GNAMES] = MULTCOMPARE(...) returns additional
%   outputs.  MEANS is a matrix with columns equal to the estimates
%   and their standard errors.  H is a handle to the figure containing
%   the graph.  GNAMES is a cell array with one row for each group,
%   containing the names of the groups.
%
%   The intervals shown in the graph are computed so that to a very close
%   approximation, two estimates being compared are significantly different
%   if their intervals are disjoint, and are not significantly different if
%   their intervals overlap.  (This is exact for multiple comparison
%   of means from anova1, if all means are based on the same sample size.)
%   You can click on any estimate to see which means are significantly
%   different from it. For Dunnett's test against a single control group,
%   that control group is selected and cannot be changed.
%
%   Two additional CriticalValueType choices are available.  The 'hsd'
%   option stands for "honestly significant differences" and is the same as
%   the 'tukey-kramer' option.  The 'lsd' option stands for "least
%   significant difference" and uses plain t-tests; it provides no
%   protection against the multiple comparison problem unless it follows a
%   preliminary overall test such as an F test.
%
%   MULTCOMPARE does not support multiple comparisons using anovan output
%   for a model that includes random or nested effects.  The calculations
%   for a random effects model produce a warning that all effects are
%   treated as fixed.  Nested models are not accepted.
%
%   Example:  Perform 1-way anova, and display group means with their names
%
%      load carsmall
%      [p,t,st] = anova1(MPG,Origin,'off');
%      [c,m,h,nms] = multcompare(st,'display','off');
%      [nms num2cell(m)]
%
%   See also ANOVA1, ANOVA2, ANOVAN, AOCTOOL, FRIEDMAN, KRUSKALWALLIS.

%   Undocumented argument for possible future use:
%     'Alternative'  Type of alternative hypothesis. Either 'notequal',
%                   'less', or 'greater'. Intended for Dunnett's test.
%                   Default is 'notequal'.


%   Also supports older calling sequence:
%      [...] = MULTCOMPARE(STATS,ALPHA,DISPLAYOPT,CTYPE,ESTIMATE,DIM)
%
%   Reference: Y. Hochberg and A.C. Tamhane, "Multiple Comparison
%   Procedures," Wiley, New York, 1987.
%
%   The Tukey-Kramer critical value is the default.  This is known
%   to be the best choice for one-way anova comparisons, but the
%   conjecture that this is best for other comparisons is
%   unproven.  The Bonferroni and Scheffe critical values are
%   conservative in all cases.

%   Copyright 1993-2022 The MathWorks, Inc.

% Check inputs and assign default values
if nargin > 1
    [varargin{:}] = convertStringsToChars(varargin{:});
end

narginchk(1,Inf);

if isempty(varargin) || (~isempty(varargin{1}) && ischar(varargin{1}))
    % Below 'ctype' is the old name, 'criticalvaluetype' is the preferred
    % new name, and 'c' is included to maintain backward compatibility so
    % that 'c' will not be ambiguous against 'control'.
    okargs =   {'alpha' 'displayopt' {'ctype' 'criticalvaluetype' 'c'} 'estimate' 'dimension' 'controlgroup' 'approximate' 'alternative'};
    defaults = {0.05    'on'         ''                                ''         []          1              []         'notequal'};
    [alpha,displayopt,ctype,estimate,dim,control,approx,alternative,isset] = ...
        internal.stats.parseArgs(okargs,defaults,varargin{:});
else
    % Old-style calling sequence with fixed argument positions
    [alpha,displayopt,ctype,estimate,dim] = getOldSyntax(varargin{:});
    control = 1;
    approx = [];
    alternative = 'notequal';
    isset.controlgroup = false;
end

if ~isstruct(stats) || ~isfield(stats,'source')
    error(message('stats:multcompare:BadStats'));
end
source = stats.source;

if (length(alpha)~=1 || ~isfloat(alpha) || ~isfinite(alpha) || alpha<=0 || alpha>=1)
    error(message('stats:multcompare:BadAlpha'));
else
    alpha = double(alpha); % do not want this to determine output type
end
if ~(isequal(displayopt, 'on') || isequal(displayopt, 'off'))
    error(message('stats:multcompare:BadDisplayOpt'));
end
if ~strcmpi(ctype,'dunnett')
    if isset.controlgroup
        error(message('stats:multcompare:NoControl'));
    else
        control = [];
    end
end
if ~isempty(alternative)
    statslib.internal.getParamVal(alternative,{'notequal' 'less' 'greater'},'''Alternative''');
    if ~strcmpi(alternative,'notequal') && ~strcmpi(ctype,'dunnett')
        error(message('stats:multcompare:NoAlternative'));
    end
else
    alternative = 'notequal';
end

dodisp = isequal(displayopt, 'on');
if isempty(approx)
    if strcmp(source,'anovan')
        approx = true;
    else
        approx = false;
    end
else
    if islogical(approx)
        validateattributes(approx,{'logical'},{'scalar'},'multcompare','Approximate');
    else
        validateattributes(approx,{'numeric'},{'scalar','integer','>=',0,'<=',1},'multcompare','Approximate');
    end
end
switch(source)
    % Calculate group means, covariances, difference, etc.
    case 'anova1'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anova1Calc(stats,ctype,alpha,control,alternative,approx);
        
    case 'anova2'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anova2Calc(stats,ctype,alpha,control,alternative,approx,estimate,dodisp);
        
    case 'anovan'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anovanCalc(stats,ctype,alpha,control,alternative,approx,dim);
        
    case 'aoctool'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = aoctoolCalc(stats,ctype,alpha,control,alternative,approx,estimate);
    
    case 'kruskalwallis'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = kruskalwallisCalc(stats,ctype,alpha,control,alternative,approx,dodisp);
        
    case 'friedman'
        [gmeans,gcov,meandiff,crit,gnames,mname,pval] = friedmanCalc(stats,ctype,alpha,control,alternative,approx,dodisp);
        
    otherwise
        error(message('stats:multcompare:BadStats'));
end

% Create output matrix showing tests for all pairwise comparisons
% and graph that approximates these tests.
[M,MM,hh] = makeM(gmeans,gcov,meandiff,crit,gnames,mname,dodisp,pval,control);

M = adjustForAlternative(M,alternative);

comparison = M;
if (nargout>1), means = MM; end
if (nargout>2), h = hh; end
end

% -----------------------------------------------
function [crit,pval] = getcrit(ctype,alpha,df,ng,t,C,alternative,approx)
% Get the minimum of the specified critical values
if (isempty(ctype))
    ctype = 'tukey-kramer';
end

crit = Inf;
if ~(ischar(ctype) || isstring(ctype))
    throwAsCaller(MException(message('stats:multcompare:BadCType')));
end

ctypes = split(ctype);
if ~isscalar(ctypes) && ismember('dunnett',ctypes)
    % Not allowed to mix dunnett with others
    throwAsCaller(MException(message('stats:multcompare:BadCType')));
end

for k = 1:length(ctypes)
    onetype = char(ctypes{k});
    if (length(onetype) == 1)
        switch onetype
            case 't', onetype = 'tukey-kramer';
            case 'd', onetype = 'dunn-sidak';
            case 'b', onetype = 'bonferroni';
            case 's', onetype = 'scheffe';
            case 'h', onetype = 'tukey-kramer';
            case 'l', onetype = 'lsd';
        end
    end
    if (isequal(onetype, 'hsd'))
        onetype = 'tukey-kramer';
    end

    
    switch onetype
        case 'tukey-kramer' % or hsd
            % The T-K algorithm is inaccurate for small alpha, so compute
            % an upper bound for it and make sure it's in range.
            ub = getcrit('dunn-sidak', alpha, df, ng, t);
            crit1 = min(ub, internal.stats.stdrinv(1-alpha, df, ng) / sqrt(2));
            pval1 = 0*t;
            for j=1:numel(t)
                pval1(j) = internal.stats.stdrcdf(sqrt(2)*abs(t(j)),df,ng,'upper');
            end
            
        case 'dunn-sidak'
            kstar = nchoosek(ng, 2);
            alf = 1-(1-alpha).^(1/kstar);
            crit1 = tinv(1-alf/2, df);
            pval1 = 1 - (1-2*tcdf(-abs(t),df)).^kstar;
            
        case 'bonferroni'
            % disp(['DF: ',num2str(ng)])
            kstar = nchoosek(ng, 2);
            kstar = 9;
            % disp(['KSTAR: ',num2str(kstar)])
            crit1 = tinv(1 - alpha / (2*kstar), df);
            pval1 = 2*kstar*tcdf(-abs(t),df);
            % disp(pval1)
            
        case 'lsd'
            crit1 = tinv(1 - alpha / 2, df);
            pval1 = 2*tcdf(-abs(t),df);
            
        case 'scheffe'
            tmp = finv(1-alpha, ng-1, df);
            crit1 = sqrt((ng-1) * tmp);
            pval1 = fcdf((t.^2)/(ng-1),ng-1,df,'upper');
            
        case 'dunnett'
            [crit1, pval1] = getcritDunnett(alpha,t,C,df,alternative,approx);
            
        otherwise
            throwAsCaller(MException(message('stats:multcompare:BadCType')));
    end
    
    pval1(pval1>1) = 1;
    if ~isfinite(crit) || crit1<crit
        crit = min(crit, crit1);
        pval = pval1;
    end
end
end

% -----------------------------------------------
function [M,MM,hh] = makeM(gmeans,gcov,meandiff,crit,gnames,mname,dodisp,pval,control)
% Create matrix to test differences, matrix of means, graph to display test

mn = meandiff.Mean;
se = meandiff.StandardError;
delta = crit * se;

ng = length(gmeans);
MM = zeros(ng,2,'like',gmeans);
MM(:,1) = gmeans(:);
MM(:,2) = sqrt(diag(gcov));
MM(isnan(MM(:,1)),2) = NaN;

M = [meandiff.Group1, meandiff.Group2, mn-delta, mn, mn+delta, pval];

% If requested, make a graph that approximates these tests
if dodisp
    hh = makeGraph(gmeans,se,crit,gnames,mname,meandiff,control);
else
    hh = [];
end
end

% -----------------------------------------------
function hh = makeGraph(gmeans,se,crit,gnames,mname,meandiff,control)
ng = length(gmeans);
i12 = sub2ind([ng ng], meandiff.Group1, meandiff.Group2);

if ~isempty(control)
    notcontrol = true(1,ng);
    notcontrol(control) = false;
    w = zeros(ng,1);
    w(notcontrol) = se;
    halfwidth = crit * w(:);
else
    % Find W values according to H&T (3.32, p. 98)
    d = zeros(ng, ng);
    d(i12) = se;
    sum1 = sum(sum(d));
    d = d + d';
    sum2 = sum(d);
    if (ng > 2)
        w = ((ng-1) * sum2 - sum1) ./ ((ng-1)*(ng-2));
    else
        w = sum1 * ones(2, 1) / 2;
    end
    halfwidth = crit * w(:);
end

hh = meansgraph(gmeans, gmeans-halfwidth, gmeans+halfwidth, ...
    gnames, mname, control);
set(hh, 'Name', getString(message('stats:multcompare:MultcompareFigureTitleString', mname)));
end

% -----------------------------------------------
function [meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,grpsize)
% Make sure NaN groups don't affect other results
t = isnan(gmeans);
if any(t)
    gcov(t,:) = 0;
    gcov(:,t) = 0;
end
ng = length(gmeans);

% Find pairs required for this comparison type
if strcmpi(ctype, "dunnett")
    g = (1:ng)';
    if ~isscalar(control) || ~isnumeric(control) || ~ismember(control,g)
        error(message('stats:multcompare:BadControl',ng));
    else
        control = double(control); % should not influence output type
    end
    g1 = g(g~=control);
    g2 = repmat(control,length(g1),1);
    C = makeDunnettC(control,grpsize,g1,gcov);
else
    M = nchoosek(1:ng, 2);      % all pairs of group numbers
    g1 = M(:,1);
    g2 = M(:,2);
    C = [];
end

% Get mean differences for these pairs
mn = gmeans(g1) - gmeans(g2);
i12 = sub2ind(size(gcov), g1, g2);
gvar = diag(gcov);
se = sqrt(gvar(g1) + gvar(g2) - 2 * gcov(i12));
t = mn./se;

% Package up for later
meandiff = struct;
meandiff.Group1 = g1;
meandiff.Group2 = g2;
meandiff.Mean = mn;
meandiff.StandardError = se;
meandiff.tValue = t;
end

% ---------------------------------------------
function C = makeDunnettC(control,grpsize,g1,gcov)
if ~isempty(grpsize)
    % All except anovan (anova1, anova2, etc.)
    ng = length(grpsize);
    N0 = grpsize(control);
    Ng = grpsize(g1);
    C = eye(ng-1);
    if ng > 1
        for i = 2:ng-1
            for j = 1:(i-1)
                C(i,j) = 1 ./ sqrt((1 + N0/Ng(i)) * (1 + N0/Ng(j)));
                C(j,i) = C(i,j);
            end
        end
    end
else
    % anovan
    n = length(g1)+1;
    idx_control = false(1,n);
    idx_control(control) = true;
    C2 = zeros(n-1, n);
    C2(:, idx_control) = -1;
    C2(:, ~idx_control) = eye(n-1);

    S = C2 * gcov * C2';
    se = (diag(S).^(1/2))';
    C = S ./ (se*se');
    pt = rank(C);
    if pt < (n-1)
        C = 0.99 .* C + 0.01 .* eye(n-1);
    end
end
end

% ------------ get paramater values from old positional syntax
function [alpha,displayopt,ctype,estimate,dim] = getOldSyntax(varargin)
% Old-style calling sequence with fixed argument positions
outargs = varargin;
n = length(outargs);
if n<5
    defaults = {0.05 'on' '' '' []};
    outargs(n+1:5) = defaults(n+1:5);
end
[alpha,displayopt,ctype,estimate,dim] = deal(outargs{:});
end

% ------------ approximate multivariate t cdf
function o = mvtcdfa(d, rho, df, p, side)
% Approximate multivariate t cdf
if df < 100
    switch side
        case 'both'
            fun = @(x,y,d,rho,df,p) 2.*(x.^(df-1).*exp(-x.^2))./(sqrt(pi)*gamma(df/2)) ...
                .* exp(-y.^2) .* ( normcdf( sqrt(2/((1-rho)*df))*d.*x + sqrt(2*rho/(1-rho)).*y ) ...
                - normcdf( - sqrt(2/((1-rho)*df))*d.*x + sqrt(2*rho/(1-rho)).*y ) ).^p;
            o = integral2(@(x,y)fun(x,y,d,rho,df,p), 0, 10, -Inf, Inf);
        case 'lower'
            fun = @(x,y,d,rho,df,p) 2.*(x.^(df-1).*exp(-x.^2))./(sqrt(pi)*gamma(df/2)) ...
                .* exp(-y.^2) .* (normcdf( sqrt(2/((1-rho)*df))*d.*x + sqrt(2*rho/(1-rho)).*y )).^p;
            o = integral2(@(x,y)fun(x,y,d,rho,df,p), 0, 10, -Inf, Inf);
        case 'upper'
            fun = @(x,y,d,rho,df,p) 2.*(x.^(df-1).*exp(-x.^2))./(sqrt(pi)*gamma(df/2)) ...
                .* exp(-y.^2) .* ( 1 ...
                - normcdf( sqrt(2/((1-rho)*df))*d.*x + sqrt(2*rho/(1-rho)).*y ) ).^p;
            o = integral2(@(x,y)fun(x,y,d,rho,df,p), 0, 10, -Inf, Inf);
    end
else
    switch side
        case 'both'
            fun = @(x,d,rho,p) exp(-(x.^2)./2)./sqrt(2*pi) ...
                .* ( normcdf( (d + sqrt(rho).*x)./sqrt(1-rho) ) ...
                - normcdf( (-d + sqrt(rho).*x)./sqrt(1-rho) ) ).^p;
            o = integral(@(x,y)fun(x,d,rho,p), -Inf, Inf);
        case 'lower'
            fun = @(x,d,rho,p) exp(-(x.^2)./2)./sqrt(2*pi) ...
                .* normcdf( (d + sqrt(rho).*x)./sqrt(1-rho) ).^p;
            o = integral(@(x,y)fun(x,d,rho,p), -Inf, Inf);
        case 'upper'
            fun = @(x,d,rho,p) exp(-(x.^2)./2)./sqrt(2*pi) ...
                .* (1 - normcdf( (d + sqrt(rho).*x)./sqrt(1-rho) )).^p;
            o = integral(@(x,y)fun(x,d,rho,p), -Inf, Inf);
    end
end
end

% ------------ get critical values and p-values for Dunnett's test
function [crit, pval] = getcritDunnett(alpha,t,C,df,alternative,approx)
p = numel(t);

if df > 1e4
    df = 1e4;
end

if p == 1
    switch alternative
        case 'notequal'
            crit = -tinv(alpha / 2, df);
            pval = 2*tcdf(-abs(t),df);
        case 'less'
            crit = tinv(alpha, df);
            pval = tcdf(t,df);
        case 'greater'
            crit = -tinv(alpha, df);
            pval = tcdf(t,df,'upper');
    end
end

if p > 1
    switch alternative
        case 'notequal'  % Ha: treatment ~= control
            crit0 = -tinv( (1-(1-alpha)^(1/p))/2, df);
            if approx
                crit = fzero(@(x)mvtcdfa(x, 0.5, df, p, 'both') ...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            else
                crit = fzero(@(x)mvtcdf(-x*ones(1,p), x*ones(1,p), C, df) ...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            end
            pval = 0*t;
            for k = 1:p
                if approx
                    pval(k) = 1 - mvtcdfa(abs(t(k)), 0.5, df, p, 'both');
                else
                    pval(k) = 1 - mvtcdf(-abs(t(k))*ones(1,p), ...
                        abs(t(k))*ones(1,p), C, df);
                end
            end
        case 'less' % Ha: treatment < control
            crit0 = tinv((1-alpha)^(1/p), df);
            if approx
                crit = fzero(@(x)mvtcdfa(x, 0.5, df, p, 'lower') ...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            else
                crit = fzero(@(x)mvtcdf(x*ones(1,p), C, df)...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            end
            pval = 0*t;
            for k = 1:p
                if approx
                    pval(k) = 1 - mvtcdfa(t(k), 0.5, df, p, 'upper');
                else
                    pval(k) = 1 - mvtcdf(t(k)*ones(1,p), Inf*ones(1,p),...
                        C, df);
                end
            end
        case 'greater' % Ha: treatment > control
            crit0 = tinv((1-alpha)^(1/p), df);
            if approx
                crit = fzero(@(x)mvtcdfa(x, 0.5, df, p, 'lower') ...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            else
                crit = fzero(@(x)mvtcdf(x*ones(1,p), C, df) ...
                    - 1 + alpha, crit0, optimset('TolX',1e-4));
            end
            pval = 0*t;
            for k = 1:p
                if approx
                    pval(k) = 1 - mvtcdfa( t(k), 0.5, df, p, 'lower');
                else
                    pval(k) = 1 - mvtcdf( t(k)*ones(1,p), C, df);
                end
            end
    end
end
end        











% ------------ calculations specific to each type of model
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anova1Calc(stats,ctype,alpha,control,alternative,approx)
mname = getString(message('stats:multcompare:mnameStringAnova1'));
gmeans = stats.means(:);
gnames = stats.gnames;
n = stats.n(:);
df = stats.df;
s = stats.s;
ng = sum(n>0);
if (df < 1)
    throwAsCaller(MException(message('stats:multcompare:NotEnoughDataANOVA')));
end

gcov = diag((s^2)./n);
[meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,n);

% Get critical value
[crit,pval] = getcrit(ctype,alpha,df,ng,meandiff.tValue,C,alternative,approx);
end

% -------------
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anova2Calc(stats,ctype,alpha,control,alternative,approx,estimate,dodisp)
docols = true;
if (~isempty(estimate))
    estimate = internal.stats.getParamVal(estimate,{'row' 'column'},'ESTIMATE');
    docols = isequal(estimate,'column');
end
if (docols)
    gmeans = stats.colmeans(:);
    n = stats.coln(:);
    mname = getString(message('stats:multcompare:mnameStringAnova2Col'));
else
    gmeans = stats.rowmeans(:);
    n = stats.rown(:);
    mname = getString(message('stats:multcompare:mnameStringAnova2Row'));
end
ng = length(gmeans);
sigma = sqrt(stats.sigmasq);
gnames = strjust(num2str((1:ng)'), 'left');
df = stats.df;
if (df < 1)
    throwAsCaller(MException(message('stats:multcompare:NotEnoughDataANOVA')));
end

gcov = ((sigma^2)/n) * eye(ng);
[meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,repmat(n,size(gmeans)));

% Get critical value
[crit,pval] = getcrit(ctype, alpha, df, ng, meandiff.tValue,C,alternative,approx);

% This whole activity is a little strange if the model includes
% interactions, especially if they are important.
if (stats.inter && dodisp)     % model included an interaction term
    if (stats.pval < alpha)
        theString = getString(message('stats:multcompare:NoteSignifInteraction'));
    else
        theString = getString(message('stats:multcompare:NoteInsignifInteraction'));
    end
    theCell = textwrap({theString},80);
    fprintf('%s\n',theCell{:});
end
end

% --------
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = anovanCalc(stats,ctype,alpha,control,alternative,approx,dim)
mname = getString(message('stats:multcompare:mnameStringAnovan'));

% We do not handle nested models
if isfield(stats,'vnested')
    vnested = stats.vnested;
else
    vnested = [];
end
if any(vnested(:))
    throwAsCaller(MException(message('stats:multcompare:NoNesting')));
end

% Our calculations treat all effects as fixed
if ~isempty(stats.denom)
    warning(message('stats:multcompare:IgnoringRandomEffects'))
end

nvars = length(stats.nlevels);
% Make sure DIM is a scalar or vector of factor numbers.
if isempty(dim)
    dim = find(stats.nlevels>1,1);
end
dim = dim(:);
if isempty(dim) || any(dim<1 | dim>nvars | dim~=round(dim))
    throwAsCaller(MException(message('stats:multcompare:BadDim', nvars)));
elseif any(strcmpi(ctype,'dunnett')) && ~isscalar(dim)
    throwAsCaller(MException(message('stats:multcompare:Dunnett1d')));
end
dim = sort(dim);
dim(diff(dim)==0) = [];
if any(stats.nlevels(dim)<2)
    throwAsCaller(MException(message('stats:multcompare:DimSpecifiesZeroDFFactor')));
end

% Create all combinations of the specified factors
ffdesign = fullfact(stats.nlevels(dim));

% Create design matrix
x = anova.utils.createANOVADesignMatrix(stats, ffdesign, dim);

% Compute estimates and their standard errors
[gmeans, gcov] = anova.utils.anovaMeanAndCovariance(stats, x);

% Get names for each group
ngroups = size(ffdesign,1);
gnames = cell(ngroups,1);
allnames = stats.grpnames;
varnames = stats.varnames;
for j=1:ngroups
    v1 = dim(1);
    vals1 = allnames{v1};
    nm = sprintf('%s=%s',varnames{v1},vals1{ffdesign(j,1)});
    for i=2:size(ffdesign,2)
        v2 = dim(i);
        vals2 = allnames{v2};
        nm = sprintf('%s,%s=%s',nm,varnames{v2},vals2{ffdesign(j,i)});
    end
    gnames{j} = nm;
end

% Get critical value
grpsize = []; % temporary
[meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,grpsize);
[crit,pval] = getcrit(ctype, alpha, stats.dfe, length(gmeans),meandiff.tValue,C,alternative,approx);

end

% --------
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = aoctoolCalc(stats,ctype,alpha,control,alternative,approx,estimate)
model = stats.model;
if (model==1 || model==3)
    throwAsCaller(MException(message('stats:multcompare:NoMultipleParameters')));
end
gnames = stats.gnames;
n = stats.n(:);
ng = length(n);
df = stats.df;
if (df < 1)
    throwAsCaller(MException(message('stats:multcompare:NotEnoughDataAOC')));
end

% Get either slope or intercept estimates and covariances
if (isempty(estimate))
    if (model == 5)
        estimate = 'slope';
    else
        estimate = 'intercept';
    end
else
    estimate = internal.stats.getParamVal(estimate,{'slope' 'intercept' 'pmm'},'ESTIMATE');
end
switch(estimate)
    case 'slope'
        if (~isfield(stats, 'slopes'))
            throwAsCaller(MException(message('stats:multcompare:BadStatsNoSlope')));
        end
        gmeans = stats.slopes;
        gcov = stats.slopecov;
        mname = getString(message('stats:multcompare:mnameStringAoctoolSlope'));
    case 'intercept'
        if (~isfield(stats, 'intercepts'))
            throwAsCaller(MException(message('stats:multcompare:BadStatsNoIntercept')));
        end
        gmeans = stats.intercepts;
        gcov = stats.intercov;
        mname = getString(message('stats:multcompare:mnameStringAoctoolIntercept'));
    case 'pmm'
        gmeans = stats.pmm;
        gcov = stats.pmmcov;
        mname = getString(message('stats:multcompare:mnameStringAoctoolPmm'));
end

if (any(any(isinf(gcov))))
    throwAsCaller(MException(message('stats:multcompare:InfiniteVariance', mname)));
end

[meandiff,C]  = makeMeanDiff(gmeans,gcov,ctype,control,n);

% Get critical value
[crit,pval] = getcrit(ctype, alpha, df, ng, meandiff.tValue,C,alternative,approx);
end

% --------
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = kruskalwallisCalc(stats,ctype,alpha,control,alternative,approx,dodisp)
gmeans = stats.meanranks(:);
gnames = stats.gnames;
n = stats.n(:);
sumt = stats.sumt;
ng = length(n);
N = sum(n);
mname = getString(message('stats:multcompare:mnameStringKruskalwallis'));

gcov = diag(((N*(N+1)/12) - (sumt/(12*(N-1)))) ./ n);
[meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,stats.n);

% Get critical value; H&T recommend the Tukey-Kramer value
[crit,pval] = getcrit(ctype, alpha, Inf, ng, meandiff.tValue,C,alternative,approx);

% Note that the intervals in M can be used for testing but not
% for simultaneous confidence intervals.  See H&T, p. 249.
if (dodisp)
    disp(getString(message('stats:multcompare:NoteNotSimul')));
end
end

% ------------
function [gmeans,gcov,meandiff,crit,gnames,mname,pval] = friedmanCalc(stats,ctype,alpha,control,alternative,approx,dodisp)
gmeans = stats.meanranks(:);
n = stats.n;
ng = length(gmeans);
sigma = stats.sigma;
mname = getString(message('stats:multcompare:mnameStringFriedman'));
gnames = strjust(num2str((1:ng)'), 'left');

gcov = ((sigma^2) / n) * eye(ng);
[meandiff,C] = makeMeanDiff(gmeans,gcov,ctype,control,repmat(stats.n,1,length(gmeans)));

% Get critical value; H&T recommend the Tukey-Kramer value
[crit,pval] = getcrit(ctype, alpha, Inf, ng, meandiff.tValue,C,alternative,approx);

% Note that the intervals in M can be used for testing but not
% for simultaneous confidence intervals.  See H&T, p. 249.
if (dodisp)
    disp(getString(message('stats:multcompare:NoteNotSimul')));
end
end

% ------------
function M = adjustForAlternative(M,alternative)
% If alternative isn't 'notequal', adjust bounds to be one-sided
switch(alternative)
    case 'less'
        M(:,3) = -Inf;
    case 'greater'
        M(:,5) = Inf;
end
end

