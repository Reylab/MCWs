wait_time = 4*3600;

while 1
    pause(wait_time)
    system("python3 wifi_error_email.py")
    % datetime('now')
end
    