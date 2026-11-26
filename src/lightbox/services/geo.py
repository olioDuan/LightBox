"""Offline bilingual gazetteer, GPS reverse geocoding, and query location parsing (zero-LLM, zero-network)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

PlaceKind = Literal["city", "region", "country"]


@dataclass(frozen=True, slots=True)
class PlaceEntry:
    """Offline geographic place entry with bilingual labels, center coordinates, and optional bbox."""

    key: str
    name_zh: str
    name_en: str
    country_key: str
    kind: PlaceKind
    lat: float
    lon: float
    radius_km: float
    aliases: tuple[str, ...]
    # Optional (min_lat, max_lat, min_lon, max_lon) for broad country / region matching
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ParsedLocationQuery:
    """Result of splitting a natural-language photo search query into location + visual keywords."""

    raw_query: str
    visual_query: str
    place: PlaceEntry | None


# Curated offline global photography travel gazetteer (cities, regions, countries) in zh-CN & en-US.
PLACES: tuple[PlaceEntry, ...] = (
    # --- Countries & Macro Regions ---
    PlaceEntry(
        "china", "中国", "China", "china", "country", 35.0, 105.0, 2400.0,
        ("中国大陆", "中国内地", "国内", "大陆", "中国", "mainland china", "china"),
        bbox=(18.0, 53.6, 73.5, 135.1),
    ),
    PlaceEntry(
        "australia", "澳大利亚", "Australia", "australia", "country", -25.3, 133.8, 2200.0,
        ("澳大利亚", "澳洲", "australia"),
        bbox=(-44.0, -10.0, 112.0, 154.5),
    ),
    PlaceEntry(
        "usa", "美国", "United States", "usa", "country", 39.8, -98.6, 2600.0,
        ("美国", "美利坚", "united states", "usa"),
        bbox=(24.3, 49.5, -125.0, -66.8),
    ),
    PlaceEntry(
        "japan", "日本", "Japan", "japan", "country", 36.2, 138.3, 1100.0,
        ("日本", "japan"),
        bbox=(24.0, 45.6, 122.9, 146.0),
    ),
    PlaceEntry(
        "korea", "韩国", "South Korea", "korea", "country", 36.5, 127.8, 350.0,
        ("韩国", "南韩", "south korea", "korea"),
        bbox=(33.0, 38.7, 124.5, 131.0),
    ),
    PlaceEntry(
        "uk", "英国", "United Kingdom", "uk", "country", 54.5, -2.5, 600.0,
        ("英国", "英格兰", "苏格兰", "united kingdom", "britain", "england", "scotland"),
        bbox=(49.8, 60.9, -8.2, 1.8),
    ),
    PlaceEntry(
        "france", "法国", "France", "france", "country", 46.6, 2.2, 600.0,
        ("法国", "france"),
        bbox=(41.3, 51.2, -5.2, 9.6),
    ),
    PlaceEntry(
        "italy", "意大利", "Italy", "italy", "country", 42.8, 12.6, 700.0,
        ("意大利", "italy"),
        bbox=(36.6, 47.1, 6.6, 18.6),
    ),
    PlaceEntry(
        "switzerland", "瑞士", "Switzerland", "switzerland", "country", 46.8, 8.2, 220.0,
        ("瑞士", "switzerland"),
        bbox=(45.8, 47.9, 5.9, 10.5),
    ),
    PlaceEntry(
        "germany", "德国", "Germany", "germany", "country", 51.2, 10.4, 500.0,
        ("德国", "germany"),
        bbox=(47.2, 55.1, 5.8, 15.1),
    ),
    PlaceEntry(
        "spain", "西班牙", "Spain", "spain", "country", 40.4, -3.7, 650.0,
        ("西班牙", "spain"),
        bbox=(35.9, 43.8, -9.4, 4.4),
    ),
    PlaceEntry(
        "iceland", "冰岛", "Iceland", "iceland", "country", 64.9, -18.9, 350.0,
        ("冰岛", "iceland"),
        bbox=(63.2, 66.6, -24.6, -13.4),
    ),
    PlaceEntry(
        "norway", "挪威", "Norway", "norway", "country", 64.5, 11.0, 900.0,
        ("挪威", "norway"),
        bbox=(57.9, 71.2, 4.5, 31.2),
    ),
    PlaceEntry(
        "greece", "希腊", "Greece", "greece", "country", 38.5, 23.7, 450.0,
        ("希腊", "greece"),
        bbox=(34.8, 41.8, 19.3, 29.7),
    ),
    PlaceEntry(
        "canada", "加拿大", "Canada", "canada", "country", 56.1, -106.3, 2800.0,
        ("加拿大", "canada"),
        bbox=(41.7, 83.1, -141.0, -52.6),
    ),
    PlaceEntry(
        "new_zealand", "新西兰", "New Zealand", "new_zealand", "country", -41.5, 172.5, 950.0,
        ("新西兰", "纽西兰", "new zealand"),
        bbox=(-47.5, -34.0, 166.0, 179.0),
    ),
    PlaceEntry(
        "thailand", "泰国", "Thailand", "thailand", "country", 13.7, 100.5, 900.0,
        ("泰国", "thailand"),
        bbox=(5.6, 20.5, 97.3, 105.7),
    ),
    PlaceEntry(
        "singapore", "新加坡", "Singapore", "singapore", "city", 1.3521, 103.8198, 45.0,
        ("新加坡", "singapore"),
    ),
    PlaceEntry(
        "indonesia", "印度尼西亚", "Indonesia", "indonesia", "country", -2.5, 118.0, 2200.0,
        ("印度尼西亚", "印尼", "indonesia"),
    ),
    PlaceEntry(
        "malaysia", "马来西亚", "Malaysia", "malaysia", "country", 4.2, 102.0, 800.0,
        ("马来西亚", "大马", "malaysia"),
    ),
    PlaceEntry(
        "vietnam", "越南", "Vietnam", "vietnam", "country", 16.0, 107.8, 950.0,
        ("越南", "vietnam"),
    ),
    PlaceEntry(
        "maldives", "马尔代夫", "Maldives", "maldives", "country", 3.2, 73.2, 450.0,
        ("马尔代夫", "maldives"),
    ),
    PlaceEntry(
        "turkey", "土耳其", "Turkey", "turkey", "country", 39.0, 35.2, 850.0,
        ("土耳其", "turkey"),
    ),
    PlaceEntry(
        "portugal", "葡萄牙", "Portugal", "portugal", "country", 39.5, -8.2, 350.0,
        ("葡萄牙", "portugal"),
    ),
    PlaceEntry(
        "netherlands", "荷兰", "Netherlands", "netherlands", "country", 52.2, 5.3, 180.0,
        ("荷兰", "netherlands", "holland"),
    ),
    PlaceEntry(
        "mexico", "墨西哥", "Mexico", "mexico", "country", 23.6, -102.5, 1500.0,
        ("墨西哥", "mexico"),
    ),
    # --- United States Cities & Regions ---
    PlaceEntry(
        "miami", "迈阿密", "Miami", "usa", "city", 25.7617, -80.1918, 95.0,
        ("迈阿密海滩", "迈阿密", "miami beach", "miami"),
    ),
    PlaceEntry(
        "florida", "佛罗里达", "Florida", "usa", "region", 27.6648, -81.5158, 450.0,
        ("佛罗里达州", "佛罗里达", "佛州", "florida"),
        bbox=(24.4, 31.0, -87.7, -79.9),
    ),
    PlaceEntry(
        "key_west", "基韦斯特", "Key West", "usa", "city", 24.5551, -81.7800, 60.0,
        ("基韦斯特", "西锁岛", "key west", "florida keys"),
    ),
    PlaceEntry(
        "orlando", "奥兰多", "Orlando", "usa", "city", 28.5383, -81.3792, 75.0,
        ("奥兰多", "orlando"),
    ),
    PlaceEntry(
        "new_york", "纽约", "New York", "usa", "city", 40.7128, -74.0060, 85.0,
        ("纽约市", "曼哈顿", "布鲁克林", "纽约", "new york city", "new york", "manhattan", "brooklyn", "nyc"),
    ),
    PlaceEntry(
        "los_angeles", "洛杉矶", "Los Angeles", "usa", "city", 34.0522, -118.2437, 95.0,
        ("洛杉矶", "好莱坞", "圣莫尼卡", "los angeles", "hollywood", "santa monica"),
    ),
    PlaceEntry(
        "san_francisco", "旧金山", "San Francisco", "usa", "city", 37.7749, -122.4194, 85.0,
        ("旧金山", "三藩市", "湾区", "san francisco", "bay area"),
    ),
    PlaceEntry(
        "california", "加州", "California", "usa", "region", 36.7783, -119.4179, 600.0,
        ("加利福尼亚", "加州", "california"),
        bbox=(32.5, 42.0, -124.5, -114.1),
    ),
    PlaceEntry(
        "las_vegas", "拉斯维加斯", "Las Vegas", "usa", "city", 36.1699, -115.1398, 75.0,
        ("拉斯维加斯", "las vegas", "vegas"),
    ),
    PlaceEntry(
        "seattle", "西雅图", "Seattle", "usa", "city", 47.6062, -122.3321, 80.0,
        ("西雅图", "seattle"),
    ),
    PlaceEntry(
        "chicago", "芝加哥", "Chicago", "usa", "city", 41.8781, -87.6298, 80.0,
        ("芝加哥", "chicago"),
    ),
    PlaceEntry(
        "boston", "波士顿", "Boston", "usa", "city", 42.3601, -71.0589, 75.0,
        ("波士顿", "boston"),
    ),
    PlaceEntry(
        "washington_dc", "华盛顿", "Washington DC", "usa", "city", 38.9072, -77.0369, 70.0,
        ("华盛顿特区", "华盛顿", "washington dc", "washington"),
    ),
    PlaceEntry(
        "san_diego", "圣地亚哥", "San Diego", "usa", "city", 32.7157, -117.1611, 75.0,
        ("圣地亚哥", "san diego"),
    ),
    PlaceEntry(
        "hawaii", "夏威夷", "Hawaii", "usa", "region", 20.7967, -156.3319, 450.0,
        ("夏威夷", "檀香山", "茂宜岛", "欧胡岛", "hawaii", "honolulu", "maui", "oahu"),
    ),
    PlaceEntry(
        "yosemite", "优胜美地", "Yosemite", "usa", "region", 37.8651, -119.5383, 75.0,
        ("优胜美地", "yosemite"),
    ),
    PlaceEntry(
        "yellowstone", "黄石", "Yellowstone", "usa", "region", 44.4280, -110.5885, 110.0,
        ("黄石公园", "黄石", "yellowstone"),
    ),
    PlaceEntry(
        "grand_canyon", "大峡谷", "Grand Canyon", "usa", "region", 36.1069, -112.1129, 90.0,
        ("科罗拉多大峡谷", "大峡谷", "grand canyon"),
    ),
    PlaceEntry(
        "alaska", "阿拉斯加", "Alaska", "usa", "region", 64.2008, -149.4937, 1100.0,
        ("阿拉斯加", "alaska"),
    ),
    # --- Australia & New Zealand Cities ---
    PlaceEntry(
        "sydney", "悉尼", "Sydney", "australia", "city", -33.8688, 151.2093, 95.0,
        ("悉尼", "雪梨", "sydney"),
    ),
    PlaceEntry(
        "melbourne", "墨尔本", "Melbourne", "australia", "city", -37.8136, 144.9631, 95.0,
        ("墨尔本", "大洋路", "melbourne", "great ocean road"),
    ),
    PlaceEntry(
        "brisbane", "布里斯班", "Brisbane", "australia", "city", -27.4698, 153.0251, 80.0,
        ("布里斯班", "brisbane"),
    ),
    PlaceEntry(
        "gold_coast", "黄金海岸", "Gold Coast", "australia", "city", -28.0167, 153.4000, 65.0,
        ("黄金海岸", "gold coast"),
    ),
    PlaceEntry(
        "perth", "珀斯", "Perth", "australia", "city", -31.9505, 115.8605, 85.0,
        ("珀斯", "perth"),
    ),
    PlaceEntry(
        "adelaide", "阿德莱德", "Adelaide", "australia", "city", -34.9285, 138.6007, 80.0,
        ("阿德莱德", "adelaide"),
    ),
    PlaceEntry(
        "cairns", "凯恩斯", "Cairns", "australia", "city", -16.9186, 145.7781, 95.0,
        ("凯恩斯", "大堡礁", "cairns", "great barrier reef"),
    ),
    PlaceEntry(
        "tasmania", "塔斯马尼亚", "Tasmania", "australia", "region", -42.0409, 146.8087, 240.0,
        ("塔斯马尼亚", "霍巴特", "tasmania", "hobart"),
    ),
    PlaceEntry(
        "canberra", "堪培拉", "Canberra", "australia", "city", -35.2809, 149.1300, 65.0,
        ("堪培拉", "canberra"),
    ),
    PlaceEntry(
        "auckland", "奥克兰", "Auckland", "new_zealand", "city", -36.8485, 174.7633, 80.0,
        ("奥克兰", "auckland"),
    ),
    PlaceEntry(
        "queenstown", "皇后镇", "Queenstown", "new_zealand", "city", -45.0312, 168.6626, 95.0,
        ("皇后镇", "瓦纳卡", "queenstown", "wanaka"),
    ),
    # --- China Cities & Regions ---
    PlaceEntry(
        "beijing", "北京", "Beijing", "china", "city", 39.9042, 116.4074, 95.0,
        ("北京市", "北京", "beijing", "peking"),
    ),
    PlaceEntry(
        "shanghai", "上海", "Shanghai", "china", "city", 31.2304, 121.4737, 85.0,
        ("上海市", "上海", "外滩", "shanghai"),
    ),
    PlaceEntry(
        "guangzhou", "广州", "Guangzhou", "china", "city", 23.1291, 113.2644, 80.0,
        ("广州市", "广州", "guangzhou", "canton"),
    ),
    PlaceEntry(
        "shenzhen", "深圳", "Shenzhen", "china", "city", 22.5431, 114.0579, 65.0,
        ("深圳市", "深圳", "shenzhen"),
    ),
    PlaceEntry(
        "hangzhou", "杭州", "Hangzhou", "china", "city", 30.2741, 120.1551, 80.0,
        ("杭州市", "杭州", "西湖", "hangzhou"),
    ),
    PlaceEntry(
        "chengdu", "成都", "Chengdu", "china", "city", 30.5728, 104.0668, 85.0,
        ("成都市", "成都", "chengdu"),
    ),
    PlaceEntry(
        "chongqing", "重庆", "Chongqing", "china", "city", 29.5630, 106.5516, 90.0,
        ("重庆市", "重庆", "chongqing"),
    ),
    PlaceEntry(
        "xian", "西安", "Xi'an", "china", "city", 34.3416, 108.9398, 80.0,
        ("西安市", "西安", "xi'an", "xian"),
    ),
    PlaceEntry(
        "nanjing", "南京", "Nanjing", "china", "city", 32.0603, 118.7969, 80.0,
        ("南京市", "南京", "nanjing"),
    ),
    PlaceEntry(
        "wuhan", "武汉", "Wuhan", "china", "city", 30.5928, 114.3055, 80.0,
        ("武汉市", "武汉", "wuhan"),
    ),
    PlaceEntry(
        "suzhou", "苏州", "Suzhou", "china", "city", 31.2989, 120.5853, 65.0,
        ("苏州市", "苏州", "suzhou"),
    ),
    PlaceEntry(
        "xiamen", "厦门", "Xiamen", "china", "city", 24.4798, 118.0894, 65.0,
        ("厦门市", "厦门", "鼓浪屿", "xiamen"),
    ),
    PlaceEntry(
        "qingdao", "青岛", "Qingdao", "china", "city", 36.0671, 120.3826, 75.0,
        ("青岛市", "青岛", "qingdao"),
    ),
    PlaceEntry(
        "dalian", "大连", "Dalian", "china", "city", 38.9140, 121.6147, 75.0,
        ("大连市", "大连", "dalian"),
    ),
    PlaceEntry(
        "sanya", "三亚", "Sanya", "china", "city", 18.2528, 109.5119, 85.0,
        ("三亚", "海口", "海南", "sanya", "hainan"),
    ),
    PlaceEntry(
        "kunming", "昆明", "Kunming", "china", "city", 24.8801, 102.8329, 85.0,
        ("昆明", "kunming"),
    ),
    PlaceEntry(
        "dali", "大理", "Dali", "china", "city", 25.6065, 100.2676, 80.0,
        ("大理", "洱海", "dali"),
    ),
    PlaceEntry(
        "lijiang", "丽江", "Lijiang", "china", "city", 26.8550, 100.2277, 85.0,
        ("丽江", "玉龙雪山", "lijiang"),
    ),
    PlaceEntry(
        "yunnan", "云南", "Yunnan", "china", "region", 25.0, 101.5, 420.0,
        ("云南省", "云南", "yunnan"),
    ),
    PlaceEntry(
        "tibet", "西藏", "Tibet", "china", "region", 29.6500, 91.1000, 650.0,
        ("西藏", "拉萨", "林芝", "tibet", "lhasa"),
    ),
    PlaceEntry(
        "xinjiang", "新疆", "Xinjiang", "china", "region", 43.8256, 87.6168, 800.0,
        ("新疆", "乌鲁木齐", "喀纳斯", "伊犁", "赛里木湖", "xinjiang", "urumqi"),
    ),
    PlaceEntry(
        "qinghai", "青海", "Qinghai", "china", "region", 36.6171, 101.7782, 480.0,
        ("青海", "西宁", "青海湖", "qinghai"),
    ),
    PlaceEntry(
        "harbin", "哈尔滨", "Harbin", "china", "city", 45.8038, 126.5350, 85.0,
        ("哈尔滨", "harbin"),
    ),
    PlaceEntry(
        "changsha", "长沙", "Changsha", "china", "city", 28.2282, 112.9388, 75.0,
        ("长沙", "changsha"),
    ),
    PlaceEntry(
        "tianjin", "天津", "Tianjin", "china", "city", 39.3434, 117.3616, 75.0,
        ("天津", "tianjin"),
    ),
    PlaceEntry(
        "hong_kong", "香港", "Hong Kong", "china", "city", 22.3193, 114.1694, 45.0,
        ("香港", "维多利亚港", "hong kong", "hongkong"),
    ),
    PlaceEntry(
        "macau", "澳门", "Macau", "china", "city", 22.1987, 113.5439, 25.0,
        ("澳门", "macau", "macao"),
    ),
    PlaceEntry(
        "taipei", "台北", "Taipei", "china", "city", 25.0330, 121.5654, 70.0,
        ("台北", "台湾", "taipei", "taiwan"),
    ),
    # --- Asia / Europe / Global Cities ---
    PlaceEntry(
        "tokyo", "东京", "Tokyo", "japan", "city", 35.6762, 139.6503, 95.0,
        ("东京都", "东京", "涩谷", "新宿", "富士山", "tokyo", "shibuya", "shinjuku", "mt fuji"),
    ),
    PlaceEntry(
        "kyoto", "京都", "Kyoto", "japan", "city", 35.0116, 135.7681, 60.0,
        ("京都", "岚山", "kyoto"),
    ),
    PlaceEntry(
        "osaka", "大阪", "Osaka", "japan", "city", 34.6937, 135.5023, 60.0,
        ("大阪", "osaka"),
    ),
    PlaceEntry(
        "hokkaido", "北海道", "Hokkaido", "japan", "region", 43.0618, 141.3545, 280.0,
        ("北海道", "札幌", "小樽", "函馆", "hokkaido", "sapporo"),
    ),
    PlaceEntry(
        "okinawa", "冲绳", "Okinawa", "japan", "region", 26.2124, 127.6809, 150.0,
        ("冲绳", "那霸", "okinawa"),
    ),
    PlaceEntry(
        "seoul", "首尔", "Seoul", "korea", "city", 37.5665, 126.9780, 75.0,
        ("首尔", "seoul"),
    ),
    PlaceEntry(
        "jeju", "济州岛", "Jeju", "korea", "region", 33.4996, 126.5312, 70.0,
        ("济州岛", "济州", "jeju"),
    ),
    PlaceEntry(
        "bangkok", "曼谷", "Bangkok", "thailand", "city", 13.7563, 100.5018, 75.0,
        ("曼谷", "bangkok"),
    ),
    PlaceEntry(
        "phuket", "普吉岛", "Phuket", "thailand", "region", 7.8804, 98.3923, 75.0,
        ("普吉岛", "普吉", "phuket"),
    ),
    PlaceEntry(
        "chiang_mai", "清迈", "Chiang Mai", "thailand", "city", 18.7883, 98.9853, 75.0,
        ("清迈", "chiang mai"),
    ),
    PlaceEntry(
        "bali", "巴厘岛", "Bali", "indonesia", "region", -8.4095, 115.1889, 100.0,
        ("巴厘岛", "峇里岛", "bali"),
    ),
    PlaceEntry(
        "paris", "巴黎", "Paris", "france", "city", 48.8566, 2.3522, 75.0,
        ("巴黎", "埃菲尔铁塔", "paris"),
    ),
    PlaceEntry(
        "london", "伦敦", "London", "uk", "city", 51.5072, -0.1276, 80.0,
        ("伦敦", "london"),
    ),
    PlaceEntry(
        "edinburgh", "爱丁堡", "Edinburgh", "uk", "city", 55.9533, -3.1883, 65.0,
        ("爱丁堡", "edinburgh"),
    ),
    PlaceEntry(
        "rome", "罗马", "Rome", "italy", "city", 41.9028, 12.4964, 75.0,
        ("罗马", "rome"),
    ),
    PlaceEntry(
        "venice", "威尼斯", "Venice", "italy", "city", 45.4408, 12.3155, 55.0,
        ("威尼斯", "venice"),
    ),
    PlaceEntry(
        "florence", "佛罗伦萨", "Florence", "italy", "city", 43.7696, 11.2558, 60.0,
        ("佛罗伦萨", "florence"),
    ),
    PlaceEntry(
        "milan", "米兰", "Milan", "italy", "city", 45.4642, 9.1900, 65.0,
        ("米兰", "milan"),
    ),
    PlaceEntry(
        "barcelona", "巴塞罗那", "Barcelona", "spain", "city", 41.3874, 2.1686, 70.0,
        ("巴塞罗那", "barcelona"),
    ),
    PlaceEntry(
        "madrid", "马德里", "Madrid", "spain", "city", 40.4168, -3.7038, 70.0,
        ("马德里", "madrid"),
    ),
    PlaceEntry(
        "lisbon", "里斯本", "Lisbon", "portugal", "city", 38.7223, -9.1393, 65.0,
        ("里斯本", "lisbon"),
    ),
    PlaceEntry(
        "amsterdam", "阿姆斯特丹", "Amsterdam", "netherlands", "city", 52.3676, 4.9041, 60.0,
        ("阿姆斯特丹", "amsterdam"),
    ),
    PlaceEntry(
        "zurich", "苏黎世", "Zurich", "switzerland", "city", 47.3769, 8.5417, 65.0,
        ("苏黎世", "zurich"),
    ),
    PlaceEntry(
        "interlaken", "因特拉肯", "Interlaken", "switzerland", "region", 46.6863, 7.8632, 65.0,
        ("因特拉肯", "少女峰", "采尔马特", "interlaken", "zermatt"),
    ),
    PlaceEntry(
        "reykjavik", "雷克雅未克", "Reykjavik", "iceland", "city", 64.1466, -21.9426, 80.0,
        ("雷克雅未克", "reykjavik"),
    ),
    PlaceEntry(
        "santorini", "圣托里尼", "Santorini", "greece", "region", 36.3932, 25.4615, 45.0,
        ("圣托里尼", "santorini"),
    ),
    PlaceEntry(
        "athens", "雅典", "Athens", "greece", "city", 37.9838, 23.7275, 65.0,
        ("雅典", "athens"),
    ),
    PlaceEntry(
        "istanbul", "伊斯坦布尔", "Istanbul", "turkey", "city", 41.0082, 28.9784, 80.0,
        ("伊斯坦布尔", "istanbul"),
    ),
    PlaceEntry(
        "dubai", "迪拜", "Dubai", "uae", "city", 25.2048, 55.2708, 90.0,
        ("迪拜", "阿布扎比", "dubai", "abu dhabi"),
    ),
    PlaceEntry(
        "vancouver", "温哥华", "Vancouver", "canada", "city", 49.2827, -123.1207, 85.0,
        ("温哥华", "vancouver"),
    ),
    PlaceEntry(
        "toronto", "多伦多", "Toronto", "canada", "city", 43.6532, -79.3832, 85.0,
        ("多伦多", "toronto"),
    ),
    PlaceEntry(
        "banff", "班夫", "Banff", "canada", "region", 51.1784, -115.5708, 120.0,
        ("班夫", "落基山", "banff"),
    ),
    PlaceEntry(
        "cancun", "坎昆", "Cancun", "mexico", "city", 21.1619, -86.8515, 85.0,
        ("坎昆", "cancun"),
    ),
)

PLACE_BY_KEY: dict[str, PlaceEntry] = {p.key: p for p in PLACES}

# Precomputed alias index sorted longest-first so "迈阿密海滩" / "miami beach" matches before "迈阿密" / "miami".
_ALIAS_INDEX: tuple[tuple[str, PlaceEntry, bool], ...] = tuple(
    sorted(
        (
            (alias.lower(), place, alias.isascii())
            for place in PLACES
            for alias in place.aliases
        ),
        key=lambda item: (-len(item[0]), item[1].kind == "country"),
    )
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points in kilometers."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _gps_in_place(lat: float, lon: float, place: PlaceEntry) -> bool:
    if place.bbox is not None and _in_bbox(lat, lon, place.bbox):
        return True
    if haversine_km(lat, lon, place.lat, place.lon) <= place.radius_km:
        return True
    if place.kind == "country":
        # Check if GPS hits any known city/region inside this country (e.g., Hawaii/Alaska for USA)
        nearest = reverse_geocode(lat, lon)
        if nearest is not None and nearest.country_key == place.key:
            return True
    return False


def reverse_geocode(lat: float, lon: float) -> PlaceEntry | None:
    """Offline nearest-city -> region -> country lookup for WGS-84 coordinates."""
    best_city: tuple[float, PlaceEntry] | None = None
    best_region: tuple[float, PlaceEntry] | None = None
    best_country: tuple[float, PlaceEntry] | None = None

    for place in PLACES:
        dist = haversine_km(lat, lon, place.lat, place.lon)
        if place.kind == "city":
            if dist <= place.radius_km and (best_city is None or dist < best_city[0]):
                best_city = (dist, place)
        elif place.kind == "region":
            inside = (place.bbox is not None and _in_bbox(lat, lon, place.bbox)) or (
                dist <= place.radius_km
            )
            if inside and (best_region is None or dist < best_region[0]):
                best_region = (dist, place)
        else:
            inside = (place.bbox is not None and _in_bbox(lat, lon, place.bbox)) or (
                dist <= place.radius_km
            )
            if inside and (best_country is None or dist < best_country[0]):
                best_country = (dist, place)

    if best_city is not None:
        return best_city[1]
    if best_region is not None:
        return best_region[1]
    if best_country is not None:
        return best_country[1]
    return None


def _find_alias_span(text: str) -> tuple[PlaceEntry, int, int] | None:
    lowered = text.lower()
    for alias, place, is_ascii in _ALIAS_INDEX:
        if is_ascii:
            m = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered)
            if m is not None:
                return place, m.start(), m.end()
        else:
            pos = lowered.find(alias)
            if pos != -1:
                return place, pos, pos + len(alias)
    return None


def match_place_in_text(text: str) -> PlaceEntry | None:
    hit = _find_alias_span(text)
    return hit[0] if hit is not None else None


def resolve_photo_location(
    gps_lat: float | None,
    gps_lon: float | None,
    session_name: str,
    rel_dir: str,
) -> tuple[PlaceEntry | None, str | None]:
    """Resolve a photo's canonical place via GPS first, falling back to session/folder name."""
    if gps_lat is not None and gps_lon is not None:
        place = reverse_geocode(gps_lat, gps_lon)
        return place, ("gps" if place is not None else None)
    combined = f"{session_name} {rel_dir}".strip()
    if combined:
        place = match_place_in_text(combined)
        if place is not None:
            return place, "folder"
    return None, None


def format_place_label(place: PlaceEntry | None, *, lang: str = "zh-CN") -> str | None:
    if place is None:
        return None
    return place.name_zh if lang.startswith("zh") else place.name_en


def parse_query_location(query: str) -> ParsedLocationQuery:
    """Detect a known place name inside a user search query and strip it to isolate visual intent."""
    raw = query.strip()
    if not raw:
        return ParsedLocationQuery(raw_query="", visual_query="", place=None)

    hit = _find_alias_span(raw)
    if hit is None:
        return ParsedLocationQuery(raw_query=raw, visual_query=raw, place=None)

    place, start, end = hit
    before = raw[:start].strip()
    after = raw[end:].strip()

    # Strip natural Chinese/English locative connectors around the place phrase
    before = re.sub(r"(?:位于|拍于|摄于|在|去|from|in|at|near|around|of)\s*$", "", before, flags=re.I).strip()
    after = re.sub(r"^(?:拍摄的|拍的|的|市|省|洲|州|附近|周边|,\s*|\s+in\b|\s+at\b)", "", after, flags=re.I).strip()
    visual = " ".join(part for part in (before, after) if part).strip(" ,，、。;；·/-")

    return ParsedLocationQuery(raw_query=raw, visual_query=visual, place=place)


def photo_matches_place(
    place: PlaceEntry,
    *,
    gps_lat: float | None,
    gps_lon: float | None,
    session_name: str,
    rel_dir: str,
) -> bool:
    """Return True if a photo matches `place` via authoritative GPS or (when GPS is absent) folder/session name."""
    if gps_lat is not None and gps_lon is not None:
        return _gps_in_place(gps_lat, gps_lon, place)

    # No GPS in EXIF: fall back to session/folder name
    folder_place = match_place_in_text(f"{session_name} {rel_dir}")
    if folder_place is None:
        return False
    if folder_place.key == place.key:
        return True
    if place.kind == "country" and folder_place.country_key == place.key:
        return True
    if place.kind == "region" and _gps_in_place(folder_place.lat, folder_place.lon, place):
        return True
    return False


__all__ = [
    "PLACES",
    "PLACE_BY_KEY",
    "ParsedLocationQuery",
    "PlaceEntry",
    "PlaceKind",
    "format_place_label",
    "haversine_km",
    "match_place_in_text",
    "parse_query_location",
    "photo_matches_place",
    "resolve_photo_location",
    "reverse_geocode",
]
