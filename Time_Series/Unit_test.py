import unittest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Define your functions within the notebook or import them properly
def load_data_from_db(engine, table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, engine)
    return df

def create_model_zone(input_layer_zone, num_zones):
    x_zone = Bidirectional(GRU(128, return_sequences=True, activation='relu'))(input_layer_zone)
    x_zone = Dropout(0.3)(x_zone)
    x_zone = Bidirectional(GRU(128, activation='relu'))(x_zone)
    x_zone = Dropout(0.3)(x_zone)
    zone_output = Dense(num_zones, activation='softmax', name='zone_output')(x_zone)
    model_zone = Model(inputs=input_layer_zone, outputs=zone_output)
    return model_zone

class TestDrivingAnalytics(unittest.TestCase):

    def setUp(self):
        # Database connection parameters 
        self.db_params = {
            'host': '194.171.191.226',
            'port': '6379',
            'database': 'postgres',
            'user': 'group20',
            'password': 'blockd_2024group20_28'
        }
        self.db_url = f"postgresql://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
        self.engine = create_engine(self.db_url)

        # Mocked data
        self.data = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, 100),
            'longitude': np.random.uniform(-180, 180, 100),
            'speed_kmh': np.random.uniform(0, 200, 100),
            'end_speed_kmh': np.random.uniform(0, 200, 100),
            'maxwaarde': np.random.uniform(0, 1, 100),
            'event_start': pd.date_range(start='1/1/2023', periods=100, freq='h'),
            'category': ['SPEED'] * 100,
            'incident_severity': ['Low'] * 50 + ['High'] * 50,
            'is_valid': [True] * 100
        })
        self.data['severity'] = self.data['incident_severity'].str.extract(r'(\d+)').fillna(0).astype(int)

    def test_load_data_from_db(self):
        # Mock the data loading function
        df = load_data_from_db(self.engine, "data_lake.safe_driving")
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_create_sequences(self):
        # Testing create_sequences function
        sample_data = np.array([[i] * 10 for i in range(20)])
        sample_target = np.array([i for i in range(20)])
        sequences, targets = create_sequences(sample_data, sample_target, 5)
        self.assertEqual(sequences.shape, (15, 5, 10))
        self.assertEqual(targets.shape, (15,))

    def test_data_preprocessing(self):
        # Test data preprocessing
        features_zone = ['latitude', 'longitude', 'speed_kmh', 'end_speed_kmh', 'maxwaarde', 'hour', 'day_of_week', 'month']
        X_zone = self.data[features_zone]
        scaler_zone = StandardScaler()
        X_scaled_zone = scaler_zone.fit_transform(X_zone)
        self.assertEqual(X_scaled_zone.shape, X_zone.shape)
    
    def test_model_creation(self):
        # Test model creation
        input_layer_zone = Input(shape=(10, 8))  # Adjust input shape as necessary
        model = create_model_zone(input_layer_zone, 5)
        self.assertIsInstance(model, Model)

    def test_model_training(self):
        # Testing model training on a small dataset
        X_sample = np.random.random((100, 10, 8))
        y_sample = to_categorical(np.random.randint(0, 5, 100), num_classes=5)
        
        input_layer_zone = Input(shape=(10, X_sample.shape[2]))
        model_zone = create_model_zone(input_layer_zone, 5)
        model_zone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model_zone.fit(X_sample, y_sample, epochs=1, batch_size=10, validation_split=0.2)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

if __name__ == '__main__':
    unittest.main()
