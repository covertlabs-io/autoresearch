from __future__ import annotations

import math
from dataclasses import dataclass


BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BITS = (16, 8, 4, 2, 1)


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class GeoBounds:
    latitude: tuple[float, float]
    longitude: tuple[float, float]

    @property
    def center(self) -> GeoPoint:
        return GeoPoint(
            latitude=(self.latitude[0] + self.latitude[1]) / 2.0,
            longitude=(self.longitude[0] + self.longitude[1]) / 2.0,
        )


def encode_geohash(latitude: float, longitude: float, precision: int = 5) -> str:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    geohash: list[str] = []
    even = True
    bit = 0
    ch = 0

    while len(geohash) < precision:
        if even:
            midpoint = (lon_range[0] + lon_range[1]) / 2.0
            if longitude > midpoint:
                ch |= BITS[bit]
                lon_range[0] = midpoint
            else:
                lon_range[1] = midpoint
        else:
            midpoint = (lat_range[0] + lat_range[1]) / 2.0
            if latitude > midpoint:
                ch |= BITS[bit]
                lat_range[0] = midpoint
            else:
                lat_range[1] = midpoint

        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def decode_geohash(geohash: str) -> GeoBounds:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    even = True

    for char in geohash.lower():
        cd = BASE32.index(char)
        for mask in BITS:
            if even:
                midpoint = (lon_range[0] + lon_range[1]) / 2.0
                if cd & mask:
                    lon_range[0] = midpoint
                else:
                    lon_range[1] = midpoint
            else:
                midpoint = (lat_range[0] + lat_range[1]) / 2.0
                if cd & mask:
                    lat_range[0] = midpoint
                else:
                    lat_range[1] = midpoint
            even = not even

    return GeoBounds(latitude=(lat_range[0], lat_range[1]), longitude=(lon_range[0], lon_range[1]))


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    lat1 = math.radians(lat_a)
    lat2 = math.radians(lat_b)
    dlat = math.radians(lat_b - lat_a)
    dlon = math.radians(lon_b - lon_a)

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius_km * c


def almost_same_cell(hash_a: str | None, hash_b: str | None, prefix: int) -> bool:
    if not hash_a or not hash_b:
        return False
    return hash_a[:prefix] == hash_b[:prefix]
