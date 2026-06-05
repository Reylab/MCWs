fprintf('Initialising audio.\n');
    
    InitializePsychSound
 [start_sound,~]=audioread('/home/user/Downloads/sound_tada_44100.wav');
    start_sound=start_sound'; %transpose to match buffer dimension

f_sample = 2*22050;

     % audiodevices = PsychPortAudio('GetDevices',2); %devicetype=2 es para Windows
%     audiodevices = PsychPortAudio('GetDevices',5); %devicetype=5 5=MacOSX/CoreAudio.
    audiodevices = PsychPortAudio('GetDevices',8); % 8=Linux/ALSA.
    outdevice = strcmp('front',{audiodevices.DeviceName});
%     outdevice = 9;

pahandle = PsychPortAudio('Open',audiodevices(outdevice).DeviceIndex,[],3,f_sample,2);
%     PTB-INFO: Using modified PortAudio V19.7.0-devel, revision 147dd722548358763a8b649b3e4b41dfffbcfbb6
PsychPortAudio('FillBuffer',pahandle,start_sound);
    PsychPortAudio('Start',pahandle);
    PsychPortAudio('Stop', pahandle,1); 

        PsychPortAudio('Close', pahandle);
