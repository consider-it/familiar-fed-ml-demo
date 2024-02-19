# Environment Variables
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Imports
import numpy as np
from cit_fl_participation.config import StorageConfig
from cit_fl_participation.participation import (
    AbstractTrainer,
    ConnectorGrpc,
    S3GlobalWeightsReader,
    S3LocalWeightsWriter,
    ParticipationServicer,
)
from model.predictor import mlModel

# Functions
def export_weights_tf(self, model):
    '''
    Converts the tf-model-weights to an 1D numpy array
    '''
    return np.concatenate([
        w.flatten()
        for w in model.get_weights()
    ])

def import_weights_tf(self, model, weights):
        '''
        Imports weights into the tf-model from a given 1D numpy array
        '''
        start = 0
        weights_list = []
        for w in model.get_weights():
            weights_list.append(weights[start:start+w.size].reshape(w.shape))
            start += w.size
        model.set_weights(weights_list)

def round_callback(model:mlModel):
    print('TODO: Setup scoring metric')

# Classes
class MyTrainer(AbstractTrainer):
    def __init__(self, predictor:mlModel, X, y, round_callback):
        super().__init__() # not absolutely necessary
        self.X = X
        self.y = y
        self.round_callback = round_callback
        self.predictor = predictor
        print("Before FL")
        round_callback(self.predictor)
    

    def train(self, import_weights, epochs):
        if import_weights.size > 0:
            import_weights_tf(self.predictor.model, import_weights)
        
        # Do some training

        round_callback(self.predictor)
        return (export_weights_tf(self.predictor.model), self.X.shape[0])
    
    def training_finished(self, final_weights):
        import_weights_tf(self.predictor.model, import_weights)
        print("Final Score")
        round_callback(self.predictor)

#### MAIN ####
def main():
    # select any token you want
    api_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # Fedml data storage
    storageConfig = StorageConfig(
        endpoint="http://49.12.108.67:9000",
        bucket="cit-fl-demo",
        access_key_id = "cit-fl-demo",
        secret_access_key = "Wy8#oS3U#$q7o40%"
    )
    reader = S3GlobalWeightsReader(storageConfig)
    writer = S3LocalWeightsWriter(storageConfig)

    # FedML server connector
    connector = ConnectorGrpc(
        heartbeat_time=1,
        coordinator_url="49.12.108.67:5051",
        api_token=api_token,
        tsl_certificate=None,
        local_weights_writer=writer,
        global_weights_reader=reader
    )

    # load training data
    X, y = np.zeros(shape=(8, 512, 512, 3)), np.zeros(shape=(8))

    # create model
    predictor = mlModel()

    # create trainer
    trainer = MyTrainer(predictor, X, y, round_callback)

    # participate using trainer
    p = ParticipationServicer(connector, trainer)
    print('Begining federated learning...')
    p.start()
    print('Finished learning')



if __name__ == "__main__":
    main()