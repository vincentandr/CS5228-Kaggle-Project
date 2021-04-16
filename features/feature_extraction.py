import time

import numpy as np
import pandas as pd
import re

# For convenient vectorized calculations of haversine distance
from haversine import haversine, haversine_vector

# Show progress bar
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

class FeatureExtraction:
    TRAIN_DATA = "../data/train.csv"
    TEST_DATA = "../data/test.csv"

    TRAIN_DATA_OUTPUT = "../data/train_output.csv"
    TEST_DATA_OUTPUT = "../data/test_output.csv"

    AUXILLARY_COMMERCIAL_CENTRES = "../data/auxiliary-data/sg-commerical-centres.csv"
    AUXILLARY_MARKETS_HAWKER_CENTRES = "../data/auxiliary-data/sg-gov-markets-hawker-centres.csv"
    AUXILLARY_POPULATION_DEMOGRAPHICS = "../data/auxiliary-data/sg-population-demographics.csv"
    AUXILLARY_PRIMARY_SCHOOLS = "../data/auxiliary-data/sg-primary-schools.csv"
    AUXILLARY_SECONDARY_SCHOOLS = "../data/auxiliary-data/sg-secondary-schools.csv"
    AUXILLARY_SHOPPING_MALLS = "../data/auxiliary-data/sg-shopping-malls.csv"
    AUXILLARY_TRAIN_STATIONS = "../data/auxiliary-data/sg-train-stations.csv"

    CPI = "../data/cpi_processed.csv"

    CBD_LATITUDE = 1.291667
    CBD_LONGITUDE = 103.85

    RADIUS_TOLERANCE = 1  # 1km radius

    def run(self):
        start_time = time.time()

        # Load data
        print("Loading data...")

        (train_data, test_data) = self.load_train_test_data()
        self.load_auxillary_data()

        # Comment out when calculating on actual data (takes a sample for testing purposes)
        # train_data = train_data.sample(n=200, random_state=42)
        # test_data = test_data.sample(n=200, random_state=42)

        # Add features to training data
        print("Adding additional features to training data...")
        train_data_new = self.add_features(train_data)

        # Add features to test data
        print("Adding additional features to test data...")
        test_data_new = self.add_features(test_data)

        # Preprocessing
        print("Preprocessing...")
        (train_data_new, test_data_new) = self.preprocess(train_data_new, test_data_new)

        print("Doing one-hot encoding...")
        train_data_new = self.one_hot_encoding(train_data_new)
        test_data_new = self.one_hot_encoding(test_data_new)

        # Save data to csv
        print("Saving data...")
        train_data_new.to_csv(self.TRAIN_DATA_OUTPUT, index=False)
        test_data_new.to_csv(self.TEST_DATA_OUTPUT, index=False)

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

        dist_to_cbd = self.get_dist_to_cbd(data)
        data['dist_to_cbd'] = dist_to_cbd

        (year, month) = self.get_month_and_year(data)
        data['year'] = year
        data['month'] = month

        (street_number_is_8, street_number_is_13, street_number_is_4) = self.get_inauspicious_street_number(data)
        data['street_number_is_8'] = street_number_is_8
        data['street_number_is_13'] = street_number_is_13
        data['street_number_is_4'] = street_number_is_4

        (block_number_is_8, block_number_is_13, block_number_is_4) = self.get_inauspicious_block_number(data)
        data['block_number_is_8'] = block_number_is_8
        data['block_number_is_13'] = block_number_is_13
        data['block_number_is_4'] = block_number_is_4

        letters = self.get_block_letter(data)
        data['block_letter'] = letters

        data['age_of_lease'] = self.get_age_of_lease(data)

        cpi = self.deflate_resale_price(data)
        data['cpi'] = cpi

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

    def get_dist_to_cbd(self, data):
        def dist_to_cbd(row):
            distance = haversine((row['latitude'], row['longitude']), (self.CBD_LATITUDE, self.CBD_LONGITUDE))
            return distance

        return data.progress_apply(dist_to_cbd, axis=1)

    def get_month_and_year(self, data):
        monthlist = data['month'].to_list()

        year, month = [], []

        for l in monthlist:
            year.append(l.split('-')[0])
            month.append(l.split('-')[1])

        return year, month

    def get_inauspicious_street_number(self, data):
        is_8 = data['street_name'].str.contains('8', regex=True).astype(int)
        is_13 = data['street_name'].str.contains('13', regex=True).astype(int)
        is_4 = data['street_name'].str.contains('4', regex=True).astype(int)

        return is_8, is_13, is_4

    def get_inauspicious_block_number(self, data):
        is_8 = data['block'].str.contains('8', regex=True).astype(int)
        is_13 = data['block'].str.contains('13', regex=True).astype(int)
        is_4 = data['block'].str.contains('4', regex=True).astype(int)

        return is_8, is_13, is_4

    def get_age_of_lease(self, data):
        return data['year'].astype(int) - data['lease_commence_date'].astype(int)

    def get_block_letter(self, data):
        letters = data['block'].str.extract('([A-Za-z])', expand=False)

        # fill in 0 to street without numbers
        letters = letters.fillna(0).to_list()

        letters_new = [ord(letter.upper()) - 64 if letter != 0 else 0 for letter in letters]

        return letters_new

    def deflate_resale_price(self, data):
        cpi = pd.read_csv(self.CPI)

        data[['year', 'month']] = data[['year', 'month']].astype(int)

        cpi[['year', 'month']] = cpi[['year', 'month']].astype(int)

        merge = data.merge(cpi, how='left', left_on=['year', 'month'], right_on=['year', 'month'])

        cpi_column = merge['cpi']

        return cpi_column

    def preprocess(self, train_data, test_data):
        # drop unused columns
        dropped_columns = ['latitude', 'longitude', 'block', 'town', 'subzone', 'street_name', 'eco_category',
                           'elevation']
        train_data_new = train_data.drop(columns=dropped_columns)
        test_data_new = test_data.drop(columns=dropped_columns)

        # drop duplicates
        train_data_new = train_data_new.drop_duplicates()

        # replace - char to space in flat type
        train_data_new['flat_type'] = train_data_new['flat_type'].str.replace('-', ' ')
        test_data_new['flat_type'] = test_data_new['flat_type'].str.replace('-', ' ')

        # flat_type convert to ordinal
        type_list = ['1 room', '2 room', '3 room', '4 room', '5 room', 'executive', 'multi generation']

        for idx, val in enumerate(type_list):
            train_data_new['flat_type'] = train_data_new['flat_type'].str.replace(val, str(idx))
            test_data_new['flat_type'] = test_data_new['flat_type'].str.replace(val, str(idx))

        # replace all values of count < min_count with 'other' in each replaced_columns
        replaced_columns = ['planning_area', 'storey_range', 'flat_model']
        min_count = 100
        for column in replaced_columns:
            column_with_counts = train_data_new[column].value_counts()
            to_be_replaced = column_with_counts[column_with_counts < min_count].index.to_list()
            train_data_new[column] = np.where(~train_data_new[column].isin(to_be_replaced),
                                              train_data_new[column], 'other_' + column)
            test_data_new[column] = np.where(~test_data_new[column].isin(to_be_replaced),
                                             test_data_new[column], 'other_' + column)

        # standardization
        excluded_columns = ['street_number_is_4', 'street_number_is_8', 'street_number_is_13', 'block_number_is_8',
                            'block_number_is_13', 'block_number_is_4', 'storey_range', 'flat_model', 'planning_area',
                            'region', 'resale_price']
        standardized_columns = list(filter(lambda x: x not in excluded_columns, list(train_data_new.columns.values)))

        scaler = StandardScaler()
        train_data_arr = scaler.fit_transform(train_data_new[standardized_columns])
        test_data_arr = scaler.transform(test_data_new[standardized_columns])

        train_data_standardized = pd.DataFrame(train_data_arr, columns=standardized_columns)
        test_data_standardized = pd.DataFrame(test_data_arr, columns=standardized_columns)

        for column in excluded_columns:
            train_data_standardized[column] = train_data_new[column].to_list()

            if column != 'resale_price':
                test_data_standardized[column] = test_data_new[column].to_list()

        # one hot encoding
        train_data_new = self.one_hot_encoding(train_data_standardized)
        test_data_new = self.one_hot_encoding(test_data_standardized)
        return train_data_new, test_data_new

    def one_hot_encoding(self, data):
        categorical_columns = ['storey_range', 'flat_model', 'planning_area', 'region']
        one_hot = [pd.get_dummies(data[column]) for column in categorical_columns]
        data = data.drop(columns=categorical_columns)
        concat_data = data
        for x in one_hot:
            concat_data = pd.concat([concat_data, x], axis=1)
        return concat_data

if __name__ == "__main__":
    tqdm.pandas()
    FeatureExtraction().run()
