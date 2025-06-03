import requests

def get_news(country, api_key='cf81aae9ae174725b4e7bc8ff9cff27f'):
    url = f'https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}'
    r = requests.get(url)
    content = r.json()
    articles = content['articles']
    results = []
    for article in articles:
        results.append(f"TITLE:\n', {article['title']},'\nDESCRIPTION\n', {article['description']}")
    return results

print("Enter your country")
country = input()
print(get_news(country)) 