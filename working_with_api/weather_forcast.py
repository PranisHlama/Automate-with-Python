import requests

# print("city,    Time,    Temperature,    Condition")
def get_weather_forcast(cityname, api_key= '3083522c86dd9b7d7d03e06fa13dd53a'):
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={cityname}&appid={api_key}'
    r = requests.get(url)
    content = r.json()
    open('data.txt', 'w').close()
    with open('data.txt', 'a') as file:
        file.write('datetime,  celcius,  condition, description \n')
        for dicty in content['list']:
            file.write(f"{dicty['dt_txt']}, {dicty['main']['temp']}, {dicty['weather'][0]['main']}, {dicty['weather'][0]['description']}\n")
        # return content
    

print(get_weather_forcast('Kathmandu'))
