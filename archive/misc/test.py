import requests

PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7897"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

cats = ["Military_radars", "Aircraft_radars", "Ground_radars", 
        "Sea_radars", "Passive_radars", "Phased_arrays",
        "Royal_Navy_Radar", "Canadian_radars", "Weather_radars",
        "Bistatic_and_multistatic_radars", "Radar_by_band"]

for cat in cats:
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action":"query","list":"categorymembers",
                "cmtitle":f"Category:{cat}","cmlimit":"3","format":"json"},
        headers=HEADERS, proxies=PROXIES, timeout=10
    )
    members = resp.json()["query"]["categorymembers"]
    print(f"{cat}: {len(members)} 个词条 {'✓' if members else '✗ 不存在'}")