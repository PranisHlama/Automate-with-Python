import requests

def get_weather_forcast(lat, lon, api= '3083522c86dd9b7d7d03e06fa13dd53a'):
    url = 'api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api}',
    r = requests.get(url)
    content = r.json()

print(get_weather_forcast('44.34', '10.99'))