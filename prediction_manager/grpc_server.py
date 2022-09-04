import os
import grpc
import logging
from concurrent import futures
from google.protobuf.json_format import MessageToDict
from prediction_manager import train_model,load_model, predict
from predictions_pb2 import IrisSpeciesPredictionResponse, CreateAndTrainModelResponse
from predictions_pb2_grpc import PredictionsServicer,add_PredictionsServicer_to_server


class Predictions(PredictionsServicer):
    def CreateAndTrainModel(self, params, context):
        params = MessageToDict(params)
        accuracy, saved_model_path = train_model(params['datasetPath'], params['epochs'],\
                             params['optimizerParams'], params['criterion'],\
                             params['modelParams'], params['saveModelAt'])
        return CreateAndTrainModelResponse(accuracy=accuracy, saved_model_path=saved_model_path)

    def IrisSpeciesPredict(self, request, context):        
        request_dict = MessageToDict(request)
        path_to_model = request_dict.pop('PathToModel')

        if os.path.exists(path_to_model): 
            model = load_model(path_to_model)
        else: 
            raise RuntimeError('Model doe not exist')
        
        predicted_species = predict(model, request_dict)

        return IrisSpeciesPredictionResponse(species=predicted_species)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PredictionsServicer_to_server(Predictions(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()