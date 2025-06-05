import requests
import json

url = 'https://api.languagetool.org/v2/check'
data = {
    'text': 'Tis is a nice Dai!',
    'language': 'auto'
}
def grammer():
    response = requests.post(url, data=data)
    result = json.loads(response.text) # converts text into dictionary
    return result

print(grammer())
