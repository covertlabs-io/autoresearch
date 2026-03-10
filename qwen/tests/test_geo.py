import unittest

from qwen_geo.geo import decode_geohash, encode_geohash, haversine_km


class GeoTests(unittest.TestCase):
    def test_geohash_round_trip_stays_close(self) -> None:
        latitude = 40.7128
        longitude = -74.0060
        geohash = encode_geohash(latitude, longitude, precision=5)
        center = decode_geohash(geohash).center
        self.assertLess(haversine_km(latitude, longitude, center.latitude, center.longitude), 5.0)

    def test_haversine_zero_for_same_point(self) -> None:
        self.assertAlmostEqual(haversine_km(10.0, 20.0, 10.0, 20.0), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
