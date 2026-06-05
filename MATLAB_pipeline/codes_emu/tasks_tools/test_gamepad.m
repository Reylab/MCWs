clear all
clear all
clear all
clc

% %%
% KbName('UnifyKeyNames');
% exitKey = KbName('escape');
% gp_state_1 = KbName('w'); %this is left key mapped with antimicro profile LR2WO
% gp_state_2 = KbName('o'); %this is button B (red) mapped with antimicro profile LR2WO
% 
% [~, ~, keyCode] = KbCheck;
% while ~keyCode(exitKey)
%     while ~any(keyCode([exitKey gp_state_1 gp_state_2]))
%         [~, ~, keyCode] = KbCheck;        
%     end
%     fprintf('Button Far Left = %d, Button Far Right = %d\n',keyCode(gp_state_1),keyCode(gp_state_2))
%     pause(0.2)
%     keyCode(gp_state_1)=0; keyCode(gp_state_2)=0;
% end
%%

KbName('UnifyKeyNames');
exitKey = KbName('escape');
disp('Press esc to exit: ');
gamepadname = 'Logitech'; %To be updated if we use a different gamepad in the future
if IsWin
    addpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX'))
    gp_state_1 = false; %this will be left key
    gp_state_2 = false; %this will be button B (red)


    clear JoyMEX
    JoyMEX('init',0);
    
    [~, ~, keyCode] = KbCheck;
    while ~keyCode(exitKey)
        while ~any([keyCode(exitKey) gp_state_1 gp_state_2])
            [~, ~, keyCode] = KbCheck; 
            [a,ab] = JoyMEX(0);
            gp_state_1 =  a(1)==-1;
            gp_state_2 =  ab(2);
        end
    %     fprintf('Button Far Left = %d, Button Far Right = %d\n',gp_state_1,gp_state_2)
        fprintf('Button Far Left = %d, Button Green = %d\n',gp_state_1,gp_state_2)
        pause(0.3)
        gp_state_1=false; gp_state_2=false;
    end
    clear JoyMEX
    rmpath(genpath('C:\Users\EMU - Behavior\Documents\MATLAB\Behavioral Tasks\HR\JoyMEX'))
elseif IsLinux

    
    numGamepads = Gamepad('GetNumGamepads');
    if (numGamepads == 0)
        error('Gamepad not connected');
    else
        [~, gamepad_name] = GetGamepadIndices;
        idx = find(contains(gamepad_name, gamepadname, 'IgnoreCase', true), 1);
        gamepad_name = gamepad_name{idx};
        gamepad_index = Gamepad('GetGamepadIndicesFromNames',gamepad_name);
        gp_numButtons = Gamepad('GetNumButtons', gamepad_index);
    end
    gp_state_1 = Gamepad('GetButton', gamepad_index, 1);
    gp_state_2 = Gamepad('GetButton', gamepad_index, 2);
    gp_state_3 = Gamepad('GetButton', gamepad_index, 3);
    gp_state_4 = Gamepad('GetButton', gamepad_index, 4);
    [~, ~, keyCode] = KbCheck;
    while ~keyCode(exitKey)
        while (~sum([gp_state_1, gp_state_2 gp_state_3 gp_state_4])) && ~keyCode(exitKey)
            gp_state_1 = Gamepad('GetButton', gamepad_index, 1);
            gp_state_2 = Gamepad('GetButton', gamepad_index, 2);
            gp_state_3 = Gamepad('GetButton', gamepad_index, 3);
            gp_state_4 = Gamepad('GetButton', gamepad_index, 4);
            [~, ~, keyCode] = KbCheck;
        end
        fprintf('Button 1 = %d, Button 2 = %d Button 3 = %d, Button 4 = %d\n',gp_state_1,gp_state_2,gp_state_3,gp_state_4)
        pause(0.1)
        gp_state_1=0; gp_state_2=0; gp_state_3=0; gp_state_4=0;
    end
else
    error('unsuported OS, check the comments')
end