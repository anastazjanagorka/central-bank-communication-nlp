import requests

url = "https://nbp.pl/wp-content/uploads/2022/09/rpp_2010_10_27.pdf"

response = requests.get(url, timeout=15)

print("Status HTTP:", response.status_code)
print("Typ pliku:", response.headers.get("Content-Type"))
print("Początek pliku:", response.content[:8])
print(response.text[:1000])