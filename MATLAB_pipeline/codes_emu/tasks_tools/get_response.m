function [gp_state_1,gp_state_2,pressed,firstPress] = get_response(dev_used,device_resp,keys,extra_pause,auto,gamepad_ix,tframe)


if ~exist('auto','var') || isempty(auto), auto=0; end
if ~exist('gamepad_ix','var') || isempty(gamepad_ix), gamepad_ix=[]; end
if ~exist('tframe','var') || isempty(tframe), tfrom=0; tout=Inf; else, tfrom=tframe(0); tout=tframe(1); end
if ~exist('extra_pause','var') || isempty(extra_pause), extra_pause=0; end
gp_state_1 = 0;
gp_state_2 = 0;
if ~auto
    gp_pressed = false;
    [pressed,firstPress]=multiKbQueueCheck(dev_used);
    if strcmp(device_resp,'gamepad')
        
        while ~any([(pressed && any(firstPress(keys))) gp_pressed]) && (GetSecs-tfrom<tout)
            [pressed,firstPress,~,~] = multiKbQueueCheck(dev_used);
            if IsWin
                [a,ab] = JoyMEX(0);
                gp_state_1 = a(1)==-1;
                gp_state_2 = ab(2);
                gp_pressed = [gp_state_1, gp_state_2];

            elseif IsLinux
                gp_state_1 = Gamepad('GetButton', gamepad_ix, 1);
                gp_state_2 = Gamepad('GetButton', gamepad_ix, 2);
                gp_pressed = [gp_state_1, gp_state_2];
            else
              %             gp_state_1 = Gamepad('GetButton', gamepad_index, 1);
               %             gp_state_2 = Gamepad('GetButton', gamepad_index, 2);
              %             gp_pressed = sum([gp_state_1, gp_state_2])
                error('not testesd in this MacOS, check commented code')
            end

        end
        pause(extra_pause)
    elseif strcmp(device_resp,'keyboard')
        while ~(pressed && any(firstPress(keys))) && (GetSecs-tfrom<tout)
            [pressed,firstPress,~,~]=multiKbQueueCheck(dev_used);
        end
    %             elseif strcmp(device_resp,'mouse')
    %                 [x,y,buttons] = GetMouse(whichScreen);
    %                 [~, ~, keyCode] = KbCheck;
    %                 while (~sum([buttons(1),buttons(2)])) && ~sum(keyCode([nextkey,exitKey]))
    %                     [x,y,buttons] = GetMouse(whichScreen);
    %                     [~, ~, keyCode] = KbCheck;
    %                 end
    %                 %             answer(ka)=buttons(1); % left person, right no person
    end
else
    pressed = 0;
    firstPress = zeros(max(keys),1);
end