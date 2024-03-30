import unittest
import json
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return '[{}]'.format(', '.join([str(obj[i]) for i in range(obj.size)]))
        # For other types, fallback to default serialization
        return super().default(obj)
    

class TestCustomJSONEncoder(unittest.TestCase):
    def test_numpy_array_serialization(self):
        # Test serialization of NumPy arrays with single dimension
        data = {
            "array1": np.array([1, 2, 3]),
            "array2": np.array([-1.5, 0, 1.5])
        }

        expected_result = {
            "array1": '[1, 2, 3]',
            "array2": '[-1.5, 0.0, 1.5]'  # Ensure floats are properly formatted
        }

        # Serialize data using custom encoder
        encoded_data = json.dumps(data, cls=CustomJSONEncoder)

        # Deserialize and compare with expected result
        decoded_data = json.loads(encoded_data)
        self.assertEqual(decoded_data, expected_result)

    def test_non_numpy_array_serialization(self):
        # Test serialization of non-NumPy arrays
        data = {
            "list1": [1, 2, 3],
            "list2": ['a', 'b', 'c']
        }

        # Ensure these arrays are serialized using default serialization
        expected_result = {
            "list1": [1, 2, 3],
            "list2": ['a', 'b', 'c']
        }

        # Serialize data using custom encoder
        encoded_data = json.dumps(data, cls=CustomJSONEncoder)

        # Deserialize and compare with expected result
        decoded_data = json.loads(encoded_data)
        self.assertEqual(decoded_data, expected_result)

    def test_multi_dimensional_array_serialization_raises_exception(self):
        # Test that the encoder raises exception when encountering multi-dimensional arrays
        data = {
            "array": np.array([[1, 2], [3, 4]])
        }

        # Attempt to serialize data using custom encoder
        with self.assertRaises(TypeError):
            json.dumps(data, cls=CustomJSONEncoder)


if __name__ == '__main__':
    unittest.main()
