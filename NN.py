'''
    Author:
    Arpit Parwal <aparwal@usc.edu>
    Yeon-soo Park <yeonsoop@usc.edu>
'''
import numpy as np
# Learning Rate
ALPHA = 0.1
# Training EPOCHS
EPOCHS = 1000
# Size of hidden layer
size_HIDDEN_LAYER = 100
# Threshold for sigmoid function
THRESHOLD = 0.5 
# Path to training set
TRAINING_FP='downgesture_train.list'
# path to test set 
TEST_FP='downgesture_test.list'

def imageToArray(pgm):
    with open(pgm, 'rb') as f:
        # skip
        f.readline()   
        f.readline()   
        # read size
        x, y = f.readline().split()
        x = int(x) 
        y = int(y)
        mScale = int(f.readline().strip())
        imgArr = []
        for _ in range(x * y):
            imgArr.append(f.read(1)[0] / mScale)
        return imgArr

class NeuralNetwork(object): 
    def __init__(self, hiddenLayerSize=size_HIDDEN_LAYER,  epoch=EPOCHS, alpha=ALPHA, 
                 trainData=None, lblTrainData=None,):       

        self.hiddenLayerSize = np.array(hiddenLayerSize)                
        self.weights = np.array
        self.alpha = alpha
        self.trainData = np.array(trainData)
        self.ipNum = 0
        self.ipDim = 0
        self.lblTrainData = np.array(lblTrainData) 
        self.opNum = 0
        self.opDim = 0
        self.X = []        
        self.layerSize = np.array  
        self.maxIter=epoch
        if (self.trainData.ndim != 0 and self.lblTrainData.ndim !=0):                                    
            self.trainData (trainData, lblTrainData)        
    # The derivation of sigmoid function.
    def deriveSigmoid(self, logX):
        return logX * (1.0-logX)   
 
    # standard squared error. 
    def errorTerm(self, x, y):
        return 2 * (x-y)
        
    def initWeights(self, layerSize):        
        weights = []
        for l in range(1, len(layerSize)):
                weights.append(((1)-(0))*np.random.normal(size=(layerSize[l-1], layerSize[l]))+(0))                   
                np.random.random
        self.weights = weights
        return self.weights
    
    # Construct the whole neural network structure, include [input layer sizes, hidden layer 1 sizes, ...hidden layer L sizes, output layer sizes]
    def setSize(self, trainData, lblTrainData):
        dim = 0
        layerSize = []
        dim = trainData.ndim
        if dim != 0:
            self.ipNum, self.ipDim = trainData.shape
        dim = lblTrainData.ndim
        if dim !=0:
            if dim == 1:
                self.output_numbers = lblTrainData.shape[0]
                self.opDim = 1
            else:
                self.output_numbers, self.opDim = lblTrainData.shape       
        layerSize.append(self.ipDim+1) # add X0        
        for i in self.hiddenLayerSize:
            layerSize.append(i)        
        layerSize.append(self.opDim) 
        self.layerSize = np.array(layerSize)
        return self.layerSize


    # run this model to get a prediction value 
    def ffn(self, input_data):
        X = [np.concatenate((np.ones(1).T, np.array(input_data)), axis=0)] 
        W = self.weights
        xj = []
        
        for l in range(0, len(W)):       
            #SIGMOID calculation  
            xj = 1.0/(1.0 + np.exp(-np.dot(X[l], W[l])))
            # Setup bias term for each hidden layer, x0=1
            if l < len(W)-1:
                xj[0] = 1 
            X.append(xj)              
        self.X = X
        return X[-1] #return the feed forward result of final level.         
    
    # In order to get the value of previous iterations and predications
    def backpropogation(self, output, label_data):
        X = self.X
        W = list(self.weights)       
        y = np.atleast_2d(label_data)   
        x = np.atleast_2d(output)
        # Base level L delta calculation.
        _ = np.average(x - y)
        _Delta = [self.errorTerm(x, y) * self.deriveSigmoid(x)] 
        # Calculate all deltas and adjust weights
        for l in range(len(X)-2, 0, -1):
            d = np.atleast_2d(_Delta[-1])
            x = np.atleast_2d(X[l])
            w = np.array(W[l])
            _Delta.append( self.deriveSigmoid(x) * _Delta[-1].dot(w.T) )    
            W[l] -= self.alpha * x.T.dot(d)
        #Calculate the weight of input layer and update weight array
        x = np.atleast_2d(X[l-1])
        d = np.atleast_2d(_Delta[-1])            
        W[l-1] -= self.alpha * x.T.dot(d)   
        self.weights = W  

    # Function to return predict array to support multiple dimension results.
    def predict(self, x):
        r = []
        r = self.ffn(x[0])        
        for i in range(len(r)):
            if r[i] >= THRESHOLD:
                r[i] = 1
            else:
                r[i] = 0
        return r
    
    # Function to train the neural network.
    def exe_trainData(self, trainData, lblTrainData):
        # print('++++  TRAINING   ++++\n')
        self.trainData = np.array(trainData)
        self.lblTrainData = np.array(lblTrainData)
        layerSize = self.setSize(self.trainData, self.lblTrainData)        
        max_iter = self.maxIter
        self.initWeights(layerSize)        
        # Execute training.
        for _ in range (0, max_iter): 
            i = np.random.randint(self.trainData.shape[0])
            _result = self.ffn(trainData[i])
            self.backpropogation(_result, lblTrainData[i])    


# Main function 
if __name__ == '__main__':
    images = []
    labels = []
    # to get the file data  
    with open(TRAINING_FP) as f:
        for training_image in f.readlines():
            training_image = training_image.strip()
            images.append(imageToArray(training_image))
            if 'down' in training_image:
                labels.append([1,])
            else:
                labels.append([0,])  
    
    nn = NeuralNetwork(hiddenLayerSize=[100,],  alpha=0.1, epoch=1000, )
    nn.exe_trainData(images, labels)
    total = 0
    correct = 0
    _dim = np.array(labels).ndim
    if _dim == 1:
        threshold_array = np.array(THRESHOLD)
    else:
        threshold_array = np.array(THRESHOLD)*np.array(labels).shape[1]
    
    with open(TEST_FP) as f:
        print('++++  PREDICTIONS   ++++\n')

        count=0
        for test_image in f.readlines():
            count+=1
            total += 1
            test_image = test_image.strip()
            p = nn.predict([imageToArray(test_image),])

            if np.all(p >= threshold_array) == ('down' in test_image):
                if np.all(p >= threshold_array) :
                    print('No.{}: {} Predict=True, Output value={}==> Match(Y)'.format(count,test_image.replace("gestures/",""), p)) 
                else:
                    print('No.{}: {} Predict=False, Output value={}==> Match(Y)'.format(count,test_image.replace("gestures/",""), p))                   
                correct += 1
            else :
                if np.all(p >= threshold_array):
                    print('No.{}: {} Predict=True, Output value={}==> Match(N)'.format(count,test_image.replace("gestures/",""), p)) 
                else:
                    print('No.{}: {} Predict=False, Output value={}==> Match(N)'.format(count,test_image.replace("gestures/",""), p)) 
    #print(nn.weights)
    print('\n++++++++\n\n')
    print('++++  SUMMARY   ++++')
    print('Total correct predictions: {}'.format(correct))
    print('Total size of data: {}'.format(total))
    print('Accuracy: {}%'.format(correct / total*100))
          
          