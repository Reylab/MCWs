function send_email_toHGR(filename,alert_subject,main_text)

alert_body = sprintf('On %s, %s led to an error %s',datetime,filename,main_text);
% alert_subject = 'gaps error';
alert_api_key = 'TAKKhX4iRO1RlfEFawH';
alert_url= "https://api.thingspeak.com/alerts/send";
jsonmessage = sprintf('{"subject": "%s", "body": "%s"}', alert_subject,alert_body);
options = weboptions("HeaderFields", {'Thingspeak-Alerts-API-Key', alert_api_key; 'Content-Type','application/json'});
try
   % webwrite(alert_url , "body", alert_body, "subject", alert_subject, options);
    webwrite(alert_url, jsonmessage, options);
catch someException
    fprintf("Failed to send alert: %s\n", someException.message);
end
