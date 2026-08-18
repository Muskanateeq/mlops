import unittest

from main import app, predict
from schemas import IrisInput


class IrisApiTests(unittest.TestCase):
    def test_application_metadata(self):
        self.assertEqual(app.title, "Iris prediction api")

    def test_prediction_returns_valid_class_and_confidence(self):
        result = predict(
            IrisInput(
                sepal_width=3.5,
                sepal_length=5.1,
                petal_width=0.2,
                petal_length=1.4,
            )
        )

        self.assertIn(result.predicted_class, {"setosa", "versicolor", "virginica"})
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)


if __name__ == "__main__":
    unittest.main()
