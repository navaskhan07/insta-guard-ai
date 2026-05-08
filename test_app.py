import unittest
import numpy as np
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

class TestInstaGuardAI(unittest.TestCase):

    def test_feature_extraction_type_casting(self):
        """UT-001: Feature extraction type casting validation"""
        profile_pic          = int('1')
        nums_length_username = float('0.27')
        fullname_words       = int('2')
        nums_length_fullname = float('0.0')
        name_eq_username     = int('0')
        description_length   = int('45')
        external_url         = int('1')
        private              = int('0')
        posts                = int('286')
        followers            = int('2740')
        follows              = int('512')

        features = np.array([[profile_pic, nums_length_username,
                              fullname_words, nums_length_fullname,
                              name_eq_username, description_length,
                              external_url, private,
                              posts, followers, follows]])

        self.assertEqual(features.shape, (1, 11))
        self.assertEqual(features[0][0], 1)
        self.assertAlmostEqual(features[0][1], 0.27)
        print("\n  UT-001 PASS: Feature extraction shape and types correct")

    def test_random_forest_inference_format(self):
        """UT-002: Random Forest inference output format"""
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
        features = np.array([[1, 0.27, 0, 0.0, 0, 32, 0, 0, 32, 1000, 500]])

        prediction = rf_model.predict(features)
        proba      = rf_model.predict_proba(features)

        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(proba.shape, (1, 2))
        self.assertAlmostEqual(sum(proba[0]), 1.0, places=5)
        print(f"\n  UT-002 PASS: RF prediction={prediction[0]}, "
              f"confidence={round(max(proba[0])*100,2)}%")

    def test_neural_network_argmax_extraction(self):
        """UT-003: Neural Network argmax and confidence extraction"""
        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')

        scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        nn_model = tf.keras.models.load_model(
                       os.path.join(MODELS_DIR, 'neural_network.keras'))

        features    = np.array([[0, 0.62, 0, 0.0, 0, 0, 0, 0, 0, 4, 102]])
        features_sc = scaler.transform(features)
        nn_pred     = nn_model.predict(features_sc, verbose=0)

        predicted_class = int(np.argmax(nn_pred[0]))
        confidence      = float(np.max(nn_pred[0]))

        self.assertEqual(nn_pred.shape, (1, 2))
        self.assertAlmostEqual(sum(nn_pred[0]), 1.0, places=5)
        self.assertIn(predicted_class, [0, 1])
        print(f"\n  UT-003 PASS: NN predicted={'FAKE' if predicted_class==1 else 'REAL'}, "
              f"confidence={round(confidence*100,2)}%")

    def test_confidence_score_computation(self):
        """UT-005: Confidence score rounding and percentage conversion"""
        test_cases = [
            (0.5,        50.0),
            (0.91666667, 91.67),
            (0.875,      87.5),
            (1.0,        100.0),
        ]
        for raw_prob, expected in test_cases:
            result = round(raw_prob * 100, 2)
            self.assertAlmostEqual(result, expected, places=1)
        print("\n  UT-005 PASS: Confidence score computation correct for all values")

if __name__ == '__main__':
    unittest.main(verbosity=2)