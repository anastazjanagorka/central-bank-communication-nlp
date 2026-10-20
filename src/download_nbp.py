import requests
url = "https://nbp.pl"

response = requests.get(url, timeout=15)

print(response.status_code)

