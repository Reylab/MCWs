function [pressed, firstPress, firstRelease, lastPress, lastRelease]  = multiKbQueueCheck(dev_use)
% This functions is equivalent to KbQueueCheck but accepts an array as
% input.
% FerChaure 2022

[pressed, firstPress, firstRelease, lastPress, lastRelease]  = KbQueueCheck(dev_use(1));


for i=2:numel(dev_use)
    [pressed2, firstPress2, firstRelease2, lastPress2, lastRelease2]  = KbQueueCheck(dev_use(i));
     pressed = pressed || pressed2;
    
    if pressed2
        for j =1:length(firstPress2)
            if firstPress2(j)~=0
                if firstPress(j)==0
                    firstPress(j)=firstPress2(j);
                else
                    firstPress(j) = min(firstPress(j),firstPress2(j));
                end
            end
            if firstRelease2(j)~=0
                if firstRelease(j)==0
                    firstRelease(j)=firstRelease2(j);
                else
                    firstRelease(j) = min(firstRelease(j),firstRelease2(j));
                end
            end

            if lastPress2(j)~=0
                if lastPress(j)==0
                    lastPress(j)=lastPress2(j);
                else
                    lastPress(j) = max(lastPress(j),lastPress2(j));
                end
            end
            
            if lastRelease2(j)~=0
                if lastRelease(j)==0
                    lastRelease(j)=lastRelease2(j);
                else
                    lastRelease(j) = max(lastRelease(j),lastRelease2(j));
                end
            end
        end
    end

end

end