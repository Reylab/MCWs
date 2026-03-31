function flickerSquare = flickerSquareLoc(windowRect,screenWidth,flickerSquareSize,RelativePos)

rectPixelsX = round((flickerSquareSize*windowRect(3))/screenWidth);
switch RelativePos
    case 'TopLeft'
        flickerSquare = [0 0 rectPixelsX rectPixelsX];
    case 'TopRight'
        flickerSquare = [windowRect(3)-rectPixelsX 0 windowRect(3) rectPixelsX];
    case 'BottomLeft'
        flickerSquare = [0 windowRect(4)-rectPixelsX rectPixelsX windowRect(4)]; 
    case 'BottomRight'
        flickerSquare = [windowRect(3)-rectPixelsX windowRect(4)-rectPixelsX windowRect(3) windowRect(4)];
end
end