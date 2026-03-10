import unittest

from qwen_geo.evaluation import parse_prediction, prediction_to_point


class ParsePredictionTests(unittest.TestCase):
    def test_parse_coords_json(self) -> None:
        parsed = parse_prediction('{"country_code":"FR","geohash":"u09t","latitude":48.8566,"longitude":2.3522}')
        self.assertEqual(parsed["country_code"], "FR")
        self.assertEqual(parsed["geohash"], "u09t")
        self.assertAlmostEqual(parsed["latitude"], 48.8566)
        self.assertAlmostEqual(parsed["longitude"], 2.3522)

    def test_parse_nested_coordinates(self) -> None:
        parsed = parse_prediction('Prediction: {"country_code":"US","coordinates":{"lat":37.7749,"lon":-122.4194}}')
        point = prediction_to_point(parsed)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point[0], 37.7749)
        self.assertAlmostEqual(point[1], -122.4194)


if __name__ == "__main__":
    unittest.main()
