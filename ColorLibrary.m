function rgb = ColorLibrary(color)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

ColorLib = struct();

% GREY
ColorLib.Grey1 = [0.8, 0.8, 0.8];
ColorLib.Grey2 = [0.6,0.6,0.6];
ColorLib.Grey3 = [0.4,0.4,0.4];
ColorLib.Grey4 = [0.2,0.2,0.2];
ColorLib.Slate = [107,113,126]./255;

% BROWNS
ColorLib.Taupe = [137,124,128]./255;
ColorLib.Sand = [222,216,203]./255;
ColorLib.CoolTaupe = [212,204,172]./255;
ColorLib.Tan = [192,176,149]./255;
ColorLib.Leather = [163,147,122]./255;
ColorLib.WarmBrown = [155,129,116]./255;
ColorLib.Brown = [126,103,75]./255;

% RED / PINK
ColorLib.Maroon = [152, 34, 34]./255;
ColorLib.Berry = [153,0,51]./255;
ColorLib.Salmon = [255,153,153]./255;
ColorLib.Cherry = [255,0,0]./255;
ColorLib.DarkRed = [204,0,0]./255;
ColorLib.DustyRose = [218, 115, 115]./255;
ColorLib.DustyRed = [191, 103, 103]./255;
ColorLib.Rose = [232,126,161]./255;
ColorLib.BrightPink = [238,38,119]./255;
ColorLib.Ballet = [235,179,169]./255;
ColorLib.RedClay = [184,99,80]./255;
ColorLib.RedBrick = [178,34,34]./255;
ColorLib.Hibiscus = [180,55,87]./255;
ColorLib.Raspberry = [210,31,60]./255;
ColorLib.RedWine = [94,25,20]./255;
ColorLib.Burgundy = [141,2,31]./255;
ColorLib.ChiliRed = [194,24,7]./255;
ColorLib.Rosewood = [136,40,40]./255;
ColorLib.Grapefruit = [201,75,75]./255;
ColorLib.Blush = [222,146,146]./255;

% ORANGE
ColorLib.Spice = [217,93,57]./255;
ColorLib.Orange = [1, 0.5, 0];
ColorLib.Melon = [255,153,0]./255;
ColorLib.Peach = [255,153,102]./255;
ColorLib.BurntOrange = [204,85,0]./255;
ColorLib.OrangeSherbet = [255, 159, 122]./255;
ColorLib.Sunset = [232,98,82]./255;
ColorLib.Tiger = [255,144,0]./255;
ColorLib.MangoOrange = [247,121,58]./255;


% YELLOW
ColorLib.Sun = [253, 216, 53]./255;
ColorLib.Golden = [240,162,2]./255;
ColorLib.Wheat = [234,186,107]./255;
ColorLib.Butterscotch = [252,177,65]./255;


% GREEN
ColorLib.DarkGreen = [0, 0.6,0.3];
ColorLib.Shamrock = [21,203,97]./255;
ColorLib.Teal = [0,153,153]./255;   
ColorLib.Kelly = [98,200,58]./255;
ColorLib.BrightGreen = [146,239,128]./255;
ColorLib.Sage = [141,170,157]./255;
ColorLib.Forest = [15,74,17]./255;
ColorLib.Pea = [76,153,0]./255;
ColorLib.Pear = [192, 202, 51]./255;
ColorLib.Fern = [124, 179, 66]./255;
ColorLib.ClassicGreen = [45, 189, 0]./255;
ColorLib.Lime = [153,194,77]./255;
ColorLib.DarkTeal = [35, 168, 154]./255;
ColorLib.GreenTea = [220,234,178]./255;
ColorLib.FadedTeal = [179,214,198]./255;
ColorLib.Pine = [33,78,52]./255;
ColorLib.Evergreen = [92,116,87]./255;
ColorLib.SpringGreen = [160,208,167]./255;
ColorLib.SlateGreen = [97,141,99]./255;
ColorLib.Olive = [191,188,89]./255;
ColorLib.Moss = [120,166,129]./255;

    
% BLUES
ColorLib.Seafoam = [4,230,171]./255;   
ColorLib.Turquoise = [51,204,204]./255;
ColorLib.Royal = [0, 0.5, 1];
ColorLib.Sky = [131,204,253]./255;
ColorLib.Cerulean = [0,110,144]./255;
ColorLib.Navy = [0,0,102]./255;
ColorLib.VibrantBlue = [0,128,255]./255;
ColorLib.CalmTeal = [105, 173, 196]./255;
ColorLib.Seafoam = [124, 232, 221]./255;
ColorLib.CalmBlue = [79, 154, 225 ]./255;
ColorLib.BrightBlue = [65,187,217]./255;
ColorLib.FadedBlue = [173,202,214]./255;
ColorLib.DarkBlue = [4,67,137]./255;
ColorLib.Aqua = [117,221,221]./255;
ColorLib.DarkSlateBlue = [78,105,129]./255;
ColorLib.LightSlateBlue = [131,157,178]./255;
ColorLib.Cornflower = [101,147,245]./255;
ColorLib.Carolina = [87,160,211]./255;
ColorLib.Lake = [69,130,180]./255;


% PURPLES
ColorLib.Violet = [153,0,204]./255;
ColorLib.Eggplant = [102,0,102]./255;
ColorLib.Periwinkle = [204,153,255]./255;
ColorLib.Purple = [147,104,183]./255;
ColorLib.DarkPeriwinkle = [105, 127, 196]./255;
ColorLib.DustyPurple = [167, 123, 206 ]./255;
ColorLib.DarkViolet = [89,84,108]./255;
ColorLib.Ink = [61,58,75]./255;
ColorLib.PurpleSlate = [146,151,196]./255;
ColorLib.ElectricPurple = [99,32,238]./255;
ColorLib.Plum = [82,43,71]./255;
 
if contains(color,'seeoptions')
    f = figure; hold on;
    f.Position(3) = f.Position(3)*2;
    
    fields = fieldnames(ColorLib);
    y = 0; yCounter = 0;
    x = 1;
    for ii = 1:length(fields)
        y = y-0.5; yCounter = yCounter+1;
        colorTitle = (fields{ii});
        rgb = ColorLib.(colorTitle);
        text(x,y,['\bf', colorTitle],'Color',rgb,'FontSize',10)
        plot(x-0.25,y,'o','MarkerEdgeColor',rgb,'MarkerFaceColor',rgb,'MarkerSize',10)
        if yCounter>=15
            y=0;
            yCounter=0;
            x = x+3;
        end
    end
    xlim([0,x+3])
    ylim([-8,0])
    rgb = [0,0,0];
else
    rgb = ColorLib.(color);
end

end























