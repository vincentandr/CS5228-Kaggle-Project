import time

import numpy as np
import pandas as pd

# For convenient vectorized calculations of haversine distance
from haversine import haversine_vector

# Show progress bar
from tqdm import tqdm


class FeatureExtraction:
    TRAIN_DATA = "./data/train.csv"
    TEST_DATA = "./data/test.csv"

    TRAIN_DATA_OUTPUT = "train_output.csv"
    TEST_DATA_OUTPUT = "test_output.csv"

    AUXILLARY_COMMERCIAL_CENTRES = "./data/auxiliary-data/sg-commerical-centres.csv"
    AUXILLARY_MARKETS_HAWKER_CENTRES = "./data/auxiliary-data/sg-gov-markets-hawker-centres.csv"
    AUXILLARY_POPULATION_DEMOGRAPHICS = "./data/auxiliary-data/sg-population-demographics.csv"
    AUXILLARY_PRIMARY_SCHOOLS = "./data/auxiliary-data/sg-primary-schools.csv"
    AUXILLARY_SECONDARY_SCHOOLS = "./data/auxiliary-data/sg-secondary-schools.csv"
    AUXILLARY_SHOPPING_MALLS = "./data/auxiliary-data/sg-shopping-malls.csv"
    AUXILLARY_TRAIN_STATIONS = "./data/auxiliary-data/sg-train-stations.csv"

    RADIUS_TOLERANCE = 1  # 1km radius

    def run(self):
        start_time = time.time()

        # Load data
        print("Loading data...")

        (train_data, test_data) = self.load_train_test_data()
        self.load_auxillary_data()

        print(train_data.shape)
        print(train_data.head())

        # Comment out when calculating on actual data (takes a sample for testing purposes)
        # train_data = train_data.sample(n=200, random_state=42)
        # test_data = test_data.sample(n=200, random_state=42)

        # Add features to training data
        print("Adding additional features to training data...")
        train_data_new = self.add_features(train_data)

        # Add features to test data
        print("Adding additional features to test data...")
        test_data_new = self.add_features(test_data)

        # Save data to csv
        print("Saving data...")
        train_data_new.to_csv(self.TRAIN_DATA_OUTPUT)
        test_data_new.to_csv(self.TEST_DATA_OUTPUT)

        # Done
        end_time = time.time()
        print(f"Completed processing in {(end_time - start_time):.3f} seconds")

    def add_features(self, data):
        dist_closest_market_hawker_centre = self.get_dist_closest_market_hawker_centre(data)
        data['dist_closest_market_hawker_centre'] = dist_closest_market_hawker_centre

        num_market_hawker_centre = self.get_num_market_hawker_centres_within_radius(data)
        data['num_market_hawker_centre_within_radius'] = num_market_hawker_centre

        dist_closest_primary_school = self.get_dist_closest_primary_school(data)
        data['dist_closest_primary_school'] = dist_closest_primary_school

        num_primary_schools = self.get_num_primary_schools_within_radius(data)
        data['num_primary_schools_within_radius'] = num_primary_schools

        dist_closest_secondary_school = self.get_dist_closest_secondary_school(data)
        data['dist_closest_secondary_school'] = dist_closest_secondary_school

        num_secondary_schools = self.get_num_secondary_schools_within_radius(data)
        data['num_secondary_schools_within_radius'] = num_secondary_schools

        dist_closest_shopping_mall = self.get_dist_closest_shopping_mall(data)
        data['dist_closest_shopping_mall'] = dist_closest_shopping_mall

        num_shopping_malls = self.get_num_shopping_malls_within_radius(data)
        data['num_shopping_malls_within_radius'] = num_shopping_malls

        dist_closest_train_station = self.get_dist_closest_train_station(data)
        data['dist_closest_train_station'] = dist_closest_train_station

        num_train_stations = self.get_num_train_stations_within_radius(data)
        data['num_train_stations_within_radius'] = num_train_stations

        return data

    def load_train_test_data(self):
        train_data = pd.read_csv(self.TRAIN_DATA)
        test_data = pd.read_csv(self.TEST_DATA)
        return (train_data, test_data)

    def load_auxillary_data(self):
        self.commercial_centres = pd.read_csv(
            self.AUXILLARY_COMMERCIAL_CENTRES)
        self.markets_hawker_centres = pd.read_csv(
            self.AUXILLARY_MARKETS_HAWKER_CENTRES)
        self.population_demographics = pd.read_csv(
            self.AUXILLARY_POPULATION_DEMOGRAPHICS)
        self.primary_schools = pd.read_csv(self.AUXILLARY_PRIMARY_SCHOOLS)
        self.secondary_schools = pd.read_csv(self.AUXILLARY_SECONDARY_SCHOOLS)
        self.shopping_malls = pd.read_csv(self.AUXILLARY_SHOPPING_MALLS)
        self.train_stations = pd.read_csv(self.AUXILLARY_TRAIN_STATIONS)

    def get_dist_closest_market_hawker_centre(self, data):
        market_hawker_centre_vector = self.markets_hawker_centres[['lat', 'lng']]

        def dist_closest_market_hawker_centre(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.markets_hawker_centres), 1))

            distance = haversine_vector(latlng_vector, market_hawker_centre_vector)
            return np.amin(distance)

        return data.progress_apply(dist_closest_market_hawker_centre, axis=1)

    def get_num_market_hawker_centres_within_radius(self, data):
        market_hawker_centre_vector = self.markets_hawker_centres[['lat', 'lng']]

        def num_market_hawker_centre_within_radius(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.markets_hawker_centres), 1))

            distance = haversine_vector(latlng_vector, market_hawker_centre_vector)
            return (distance <= self.RADIUS_TOLERANCE).sum()

        return data.progress_apply(num_market_hawker_centre_within_radius, axis=1)

    def get_dist_closest_primary_school(self, data):
        primary_schools_vector = self.primary_schools[['lat', 'lng']]

        def dist_closest_primary_school(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.primary_schools), 1))

            distance = haversine_vector(latlng_vector, primary_schools_vector)
            return np.amin(distance)

        return data.progress_apply(dist_closest_primary_school, axis=1)

    def get_num_primary_schools_within_radius(self, data):
        primary_schools_vector = self.primary_schools[['lat', 'lng']]

        def num_primary_schools_within_radius(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.primary_schools), 1))

            distance = haversine_vector(latlng_vector, primary_schools_vector)
            return (distance <= self.RADIUS_TOLERANCE).sum()

        return data.progress_apply(num_primary_schools_within_radius, axis=1)

    def get_dist_closest_secondary_school(self, data):
        secondary_schools_vector = self.secondary_schools[['lat', 'lng']]

        def dist_closest_secondary_school(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.secondary_schools), 1))

            distance = haversine_vector(latlng_vector, secondary_schools_vector)
            return np.amin(distance)

        return data.progress_apply(dist_closest_secondary_school, axis=1)

    def get_num_secondary_schools_within_radius(self, data):
        secondary_schools_vector = self.secondary_schools[['lat', 'lng']]

        def num_secondary_schools_within_radius(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.secondary_schools), 1))

            distance = haversine_vector(latlng_vector, secondary_schools_vector)
            return (distance <= self.RADIUS_TOLERANCE).sum()

        return data.progress_apply(num_secondary_schools_within_radius, axis=1)

    def get_dist_closest_shopping_mall(self, data):
        shopping_malls_vector = self.shopping_malls[['lat', 'lng']]

        def dist_closest_shopping_mall(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.shopping_malls), 1))

            distance = haversine_vector(latlng_vector, shopping_malls_vector)
            return np.amin(distance)

        return data.progress_apply(dist_closest_shopping_mall, axis=1)

    def get_num_shopping_malls_within_radius(self, data):
        shopping_malls_vector = self.shopping_malls[['lat', 'lng']]

        def num_shopping_malls_within_radius(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.shopping_malls), 1))

            distance = haversine_vector(latlng_vector, shopping_malls_vector)
            return (distance <= self.RADIUS_TOLERANCE).sum()

        return data.progress_apply(num_shopping_malls_within_radius, axis=1)

    def get_dist_closest_train_station(self, data):
        train_stations_vector = self.train_stations[['lat', 'lng']]

        def dist_closest_train_station(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.train_stations), 1))

            distance = haversine_vector(latlng_vector, train_stations_vector)
            return np.amin(distance)

        return data.progress_apply(dist_closest_train_station, axis=1)

    def get_num_train_stations_within_radius(self, data):
        train_stations_vector = self.train_stations[['lat', 'lng']]

        def num_train_stations_within_radius(row):
            latlng_vector = np.tile(tuple(row[['latitude', 'longitude']]),
                                    (len(self.train_stations), 1))

            distance = haversine_vector(latlng_vector, train_stations_vector)
            return (distance <= self.RADIUS_TOLERANCE).sum()

        return data.progress_apply(num_train_stations_within_radius, axis=1)


if __name__ == "__main__":
    tqdm.pandas()
    FeatureExtraction().run()
