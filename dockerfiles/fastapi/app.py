import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal, List
from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated
from enum import Enum

# Valid categories based on training data
class PropertyType(str, Enum):
    BUS = "Bus"
    CAMPER_RV = "Camper/RV"
    CASA_PARTICULAR = "Casa particular"
    CASTLE = "Castle"
    CAVE = "Cave"
    ENTIRE_BUNGALOW = "Entire bungalow"
    ENTIRE_CABIN = "Entire cabin"
    ENTIRE_CHALET = "Entire chalet"
    ENTIRE_CONDO = "Entire condo"
    ENTIRE_COTTAGE = "Entire cottage"
    ENTIRE_GUEST_SUITE = "Entire guest suite"
    ENTIRE_GUESTHOUSE = "Entire guesthouse"
    ENTIRE_HOME = "Entire home"
    ENTIRE_HOME_APT = "Entire home/apt"
    ENTIRE_LOFT = "Entire loft"
    ENTIRE_PLACE = "Entire place"
    ENTIRE_RENTAL_UNIT = "Entire rental unit"
    ENTIRE_SERVICED_APARTMENT = "Entire serviced apartment"
    ENTIRE_TOWNHOUSE = "Entire townhouse"
    ENTIRE_VACATION_HOME = "Entire vacation home"
    ENTIRE_VILLA = "Entire villa"
    PENSION = "Pension"
    PRIVATE_ROOM = "Private room"
    PRIVATE_ROOM_BB = "Private room in bed and breakfast"
    PRIVATE_ROOM_CASA = "Private room in casa particular"
    PRIVATE_ROOM_CASTLE = "Private room in castle"
    PRIVATE_ROOM_CHALET = "Private room in chalet"
    PRIVATE_ROOM_CONDO = "Private room in condo"
    PRIVATE_ROOM_GUEST_SUITE = "Private room in guest suite"
    PRIVATE_ROOM_GUESTHOUSE = "Private room in guesthouse"
    PRIVATE_ROOM_HOME = "Private room in home"
    PRIVATE_ROOM_HOSTEL = "Private room in hostel"
    PRIVATE_ROOM_LOFT = "Private room in loft"
    PRIVATE_ROOM_RENTAL = "Private room in rental unit"
    PRIVATE_ROOM_RESORT = "Private room in resort"
    PRIVATE_ROOM_SERVICED = "Private room in serviced apartment"
    PRIVATE_ROOM_TINY = "Private room in tiny home"
    PRIVATE_ROOM_TOWNHOUSE = "Private room in townhouse"
    PRIVATE_ROOM_VACATION = "Private room in vacation home"
    PRIVATE_ROOM_VILLA = "Private room in villa"
    ROOM_APARTHOTEL = "Room in aparthotel"
    ROOM_BB = "Room in bed and breakfast"
    ROOM_BOUTIQUE = "Room in boutique hotel"
    ROOM_HOSTEL = "Room in hostel"
    ROOM_HOTEL = "Room in hotel"
    ROOM_SERVICED = "Room in serviced apartment"
    SHARED_ROOM = "Shared room"
    SHARED_ROOM_BARN = "Shared room in barn"
    SHARED_ROOM_BB = "Shared room in bed and breakfast"
    SHARED_ROOM_CASA = "Shared room in casa particular"
    SHARED_ROOM_CONDO = "Shared room in condo"
    SHARED_ROOM_GUESTHOUSE = "Shared room in guesthouse"
    SHARED_ROOM_HOME = "Shared room in home"
    SHARED_ROOM_HOSTEL = "Shared room in hostel"
    SHARED_ROOM_HOTEL = "Shared room in hotel"
    SHARED_ROOM_LOFT = "Shared room in loft"
    SHARED_ROOM_RENTAL = "Shared room in rental unit"
    SHARED_ROOM_SERVICED = "Shared room in serviced apartment"
    SHARED_ROOM_TENT = "Shared room in tent"
    SHARED_ROOM_TOWNHOUSE = "Shared room in townhouse"
    SHARED_ROOM_VILLA = "Shared room in villa"
    TINY_HOME = "Tiny home"
    TOWER = "Tower"

class RoomType(str, Enum):
    ENTIRE_HOME_APT = "Entire home/apt"
    HOTEL_ROOM = "Hotel room"
    PRIVATE_ROOM = "Private room"
    SHARED_ROOM = "Shared room"

class City(str, Enum):
    BUENOS_AIRES = "Buenos Aires"


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Skip MLflow entirely if running locally (no Docker environment)
        import os
        if not os.path.exists('/.dockerenv'):
            print("Local environment detected, skipping MLflow...")
            raise Exception("Local environment - skip MLflow")
            
        # Load the trained model from MLflow with quick timeout
        print("Attempting MLflow connection...")
        mlflow.set_tracking_uri('http://mlflow:5000')
        
        # Quick connectivity test first (2 second timeout)
        import requests
        try:
            response = requests.get('http://mlflow:5000', timeout=2)
            print("MLflow server is reachable")
        except:
            print("MLflow server not reachable (timeout), using local model")
            raise Exception("MLflow server not available")
        
        client_mlflow = mlflow.MlflowClient()
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
        print(f"Loaded model from MLflow: version {version_model_ml}")
    except Exception as e:
        # If there is no registry in MLflow, open the default model
        print(f"MLflow loading failed ({e}), loading local model...")
        import os
        model_path = './files/model.pkl' if os.path.exists('./files/model.pkl') else '/app/files/model.pkl'
        try:
            file_ml = open(model_path, 'rb')
            model_ml = pickle.load(file_ml)
            file_ml.close()
            version_model_ml = 0
            print(f"Loaded local model from {model_path}")
        except Exception as local_error:
            print(f"Error loading local model: {local_error}")
            raise local_error

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
        print("Loaded data dictionary from S3")
    except Exception as e:
        # If data dictionary is not found in S3, load it from local file
        print(f"S3 loading failed ({e}), loading local data dictionary...")
        import os
        data_path = './files/data.json' if os.path.exists('./files/data.json') else '/app/files/data.json'
        try:
            file_s3 = open(data_path, 'r')
            data_dictionary = json.load(file_s3)
            file_s3.close()
            print(f"Loaded local data dictionary from {data_path}")
        except Exception as local_error:
            print(f"Error loading local data dictionary: {local_error}")
            raise local_error

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "airbnb_occupancy_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)
            print(f"Model updated to version {version_model}")

    except Exception as e:
        # If an error occurs during the process, log it but don't crash
        print(f"Error checking model updates: {e}")
        pass
 

class AirbnbFeatures(BaseModel):
    """
    Input schema for Airbnb property features based on training data ranges and categories
    """
    
    # Geographic coordinates (Buenos Aires area)
    latitude: float = Field(
        description="Latitude coordinate", 
        ge=-34.68, 
        le=-34.53,
        example=-34.6037
    )
    longitude: float = Field(
        description="Longitude coordinate", 
        ge=-58.53, 
        le=-58.36,
        example=-58.3816
    )
    
    # Property characteristics
    property_type: PropertyType = Field(
        description="Type of property",
        example="Entire rental unit"
    )
    room_type: RoomType = Field(
        description="Type of room",
        example="Entire home/apt"
    )
    
    # Capacity and space
    accommodates: int = Field(
        description="Number of guests the property accommodates", 
        ge=1, 
        le=10,
        example=4
    )
    bedrooms: float = Field(
        description="Number of bedrooms", 
        ge=0.0, 
        le=10.0,
        example=2.0
    )
    beds: float = Field(
        description="Number of beds", 
        ge=0.0, 
        le=10.0,
        example=2.0
    )
    
    # Pricing and booking policies
    price: float = Field(
        description="Price per night in USD", 
        ge=0.23, 
        le=90927.88,
        example=150.0
    )
    minimum_nights: int = Field(
        description="Minimum number of nights required", 
        ge=1, 
        le=730,
        example=2
    )
    maximum_nights: int = Field(
        description="Maximum number of nights allowed", 
        ge=1, 
        le=99999,
        example=30
    )
    
    # Reviews and ratings
    number_of_reviews: int = Field(
        description="Total number of reviews", 
        ge=0, 
        le=992,
        example=45
    )
    review_scores_rating: float = Field(
        description="Overall review score rating", 
        ge=0.0, 
        le=5.0,
        example=4.8
    )
    review_scores_accuracy: float = Field(
        description="Review score for accuracy", 
        ge=0.0, 
        le=5.0,
        example=4.9
    )
    review_scores_cleanliness: float = Field(
        description="Review score for cleanliness", 
        ge=0.0, 
        le=5.0,
        example=4.7
    )
    review_scores_checkin: float = Field(
        description="Review score for check-in process", 
        ge=0.0, 
        le=5.0,
        example=4.8
    )
    review_scores_communication: float = Field(
        description="Review score for communication", 
        ge=1.0, 
        le=5.0,
        example=4.9
    )
    review_scores_location: float = Field(
        description="Review score for location", 
        ge=1.0, 
        le=5.0,
        example=4.6
    )
    review_scores_value: float = Field(
        description="Review score for value", 
        ge=1.0, 
        le=5.0,
        example=4.5
    )
    
    # Host information
    host_listings_count: float = Field(
        description="Number of listings the host has", 
        ge=1.0, 
        le=670.0,
        example=3.0
    )
    host_total_listings_count: float = Field(
        description="Total number of listings the host has ever had", 
        ge=1.0, 
        le=2542.0,
        example=3.0
    )
    
    # Binary features
    instant_bookable: int = Field(
        description="Whether the property is instantly bookable (0=No, 1=Yes)", 
        ge=0, 
        le=1,
        example=1
    )
    host_is_superhost: int = Field(
        description="Whether the host is a superhost (0=No, 1=Yes)", 
        ge=0, 
        le=1,
        example=0
    )
    
    # Location
    city: City = Field(
        description="City where the property is located",
        example="Buenos Aires"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": -34.6037,
                    "longitude": -58.3816,
                    "property_type": "Entire rental unit",
                    "room_type": "Entire home/apt",
                    "accommodates": 4,
                    "bedrooms": 2.0,
                    "beds": 2.0,
                    "price": 150.0,
                    "minimum_nights": 2,
                    "maximum_nights": 30,
                    "number_of_reviews": 45,
                    "review_scores_rating": 4.8,
                    "review_scores_accuracy": 4.9,
                    "review_scores_cleanliness": 4.7,
                    "review_scores_checkin": 4.8,
                    "review_scores_communication": 4.9,
                    "review_scores_location": 4.6,
                    "review_scores_value": 4.5,
                    "instant_bookable": 1,
                    "host_is_superhost": 0,
                    "host_listings_count": 3.0,
                    "host_total_listings_count": 3.0,
                    "city": "Buenos Aires"
                }
            ]
        }
    }


class ModelInput(BaseModel):
    """
    Input schema for the Airbnb occupancy prediction model.

    This class defines the input structure that accepts Airbnb property features.
    """

    features: List[AirbnbFeatures] = Field(
        description="List of Airbnb property features for prediction",
        min_length=1,
        max_length=1,
    )


class ModelOutput(BaseModel):
    """
    Output schema for the Airbnb occupancy prediction model.

    This class defines the output fields returned by the occupancy prediction model.
    
    :param prediction: Prediction for the property. 1 for high occupancy, 0 for low occupancy.
    :param occupancy_level: Descriptive label for the occupancy prediction.
    """

    prediction: int = Field(
        description="Prediction for the property. 1 for high occupancy, 0 for low occupancy",
        ge=0,
        le=1,
    )
    occupancy_level: str = Field(
        description="Descriptive label for the occupancy level",
        examples=["Low Occupancy", "High Occupancy"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 1
                }
            ]
        }
    }


# Load the model before start
print("Starting model loading...")
model, version_model, data_dict = load_model("airbnb_occupancy_model_prod", "champion")
print("Model loading completed!")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Airbnb Occupancy Prediction API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Airbnb Occupancy Prediction API for Buenos Aires"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    input_data: ModelInput,
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting Airbnb occupancy in Buenos Aires.

    This endpoint receives features related to an Airbnb property and predicts whether the property 
    will have high occupancy (1) or low occupancy (0) using a trained model.
    """

    try:
        # Get the single property from the input (only one property allowed)
        property_features = input_data.features[0]
        
        # Extract features from the request and convert them into a DataFrame
        features_dict = property_features.dict()
        
        # Create DataFrame with the correct column order
        features_df = pd.DataFrame([features_dict])
        
        # Ensure column order matches training data
        if "columns" in data_dict:
            # Reorder columns to match training data
            features_df = features_df[data_dict["columns"]]
        
        # Make the prediction using the trained model (preprocessing is handled by the pipeline)
        prediction = model.predict(features_df)
        
        # Determine occupancy level description
        prediction_value = int(prediction[0])
        occupancy_level = "High Occupancy" if prediction_value == 1 else "Low Occupancy"

        # Check if the model has changed asynchronously
        background_tasks.add_task(check_model)

        # Return the prediction result
        return ModelOutput(prediction=prediction_value, occupancy_level=occupancy_level)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return a default prediction in case of error
        return ModelOutput(prediction=0, occupancy_level="Low Occupancy")
 