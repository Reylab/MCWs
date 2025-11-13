import requests

api_key = 'bbW4hPFGU9DwEv6Yuc6hei'
event_name = 'check_wifi'

url = f'https://maker.ifttt.com/trigger/{event_name}/with/key/{api_key}'
response = requests.post(url)
